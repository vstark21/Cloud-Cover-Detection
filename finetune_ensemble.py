# General imports
from collections import defaultdict
import os
import cv2
import time
import glob
import json
import yaml
import scipy
import random
import warnings
import datetime
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as A
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

import models
from datasets import CloudDataset
from losses import CloudLoss, jaccard_loss
from utils import *
from loguru import logger

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_ids', dest='model_ids', type=str, help='Model ids', default=None)
args = parser.parse_args()

# config
with open("config.yml", "r") as f:
    config = AttrDict(yaml.safe_load(f))
if config.DEBUG:
    config.USE_WANDB = False
    config.AMP = False
    config.N_ACCUMULATE = 2

config.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if not os.path.exists(config.OUTPUT_PATH):
    os.makedirs(config.OUTPUT_PATH)
logger.info(f"Using device: {config.DEVICE}")
if config.USE_WANDB:
    import wandb
    wandb.init(
        project=config.NAME, 
        entity="vstark21", 
        config=config
    )
torch.autograd.set_detect_anomaly(config.DEBUG)

if __name__ == "__main__":
    files = []
    for name in glob.glob(
        os.path.join(config.DATA_PATH, "*.npz")
    ):
        chip_id = os.path.split(name)[-1].split(".")[0]
        if 'label' in chip_id:
            continue    
        files.append({
            "chip_id": chip_id,
            "feat_path": name,
            "label_path": name.replace(".npz", "_label.npz")
        })
    logger.info("Ensemble Finetuning with {} chips".format(
        len(files)
    ))
    dataset = CloudDataset(
        files, config.DATA_MEAN, config.DATA_STD, use_bands=config.USE_BANDS)
    
    dataloader = DataLoader(
                        dataset,
                        batch_size=config.TRAIN_BATCH_SIZE,
                        shuffle=True,
                        num_workers=config.NUM_WORKERS,
                        worker_init_fn=worker_init_fn)

    loss_fn = CloudLoss(
        config.LOSS_CFG
    )

    ensemble_models = []
    if args.model_ids is None:
        raise ValueError("Please provide model ids")
    for k in args.model_ids.split(","):
        v = config.FT_MODELS[k]
        logger.info(f"Loading checkpoint: {k}")
        _model = getattr(models, v['MODEL'])(
            **v['MODEL_PARAMS']
            ).to(config.DEVICE)
        _model.load_state_dict(
            torch.load(os.path.join(config.FT_MODELS_DIR, f"{k}.pt"), map_location=config.DEVICE)
        )
        _model.eval()
        ensemble_models.append(_model)
    ensembler = getattr(models, config.FT_ENSEMBLER)(
        in_channels=len(ensemble_models), out_channels=1
    ).to(config.DEVICE)
    optimizer = getattr(torch.optim, config.FT_OPTIMIZER_CFG.pop('type'))(
        ensembler.parameters(), **config.FT_OPTIMIZER_CFG
    )
    grad_scaler = torch.cuda.amp.GradScaler(enabled=config.AMP)

    logger.info(f"Ensembler has {count_parameters(ensembler)} parameters")
    for epoch in range(config.FT_EPOCHS):
        tic = time.time()

        # Finetuning
        ensembler.train()
        ensembler.zero_grad()
        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        metrics = defaultdict(lambda: 0)
        dataset_len = 0

        for step, batch_data in bar:

            images = batch_data['inputs'].to(config.DEVICE)
            labels = batch_data['labels'].to(config.DEVICE)
            
            with torch.cuda.amp.autocast(enabled=config.AMP):
                with torch.no_grad():
                    images = torch.cat([
                        torch.sigmoid(model(images)['out']) for model in ensemble_models
                    ], dim=1)
                preds_dict = ensembler(images)
                loss_dict = loss_fn(preds_dict, labels)
                loss = loss_dict['loss']

                for name, value in loss_dict.items():
                    metrics[name] += value.item()

                metrics['jaccard'] += jaccard_score(preds_dict['out'], labels).item()

                loss = loss / config.N_ACCUMULATE

            grad_scaler.scale(loss).backward()

            if (step + 1) % config.N_ACCUMULATE == 0:
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad()

            dataset_len += 1

            bar.set_postfix(
                epoch=epoch, 
                jac_loss=metrics['jacc_loss'] / dataset_len,
                jaccard=metrics['jaccard'] / dataset_len,
                # **{name: value / dataset_len 
                #             for name, value in metrics.items()},
                lr=optimizer.param_groups[0]['lr']
            )
        metrics = {
            name: value / dataset_len for name, value in metrics.items()
        }
        for name, value in metrics.items():
            logger.info(f"{name}: {value}")

        if config.USE_WANDB:
            log_dict = dict()
            for name, value in metrics.items():
                log_dict[f"{name}"] = value
            log_dict['lr'] = optimizer.param_groups[0]['lr']
            wandb.log(log_dict)

        logger.info(f"Epoch {epoch} ended, time taken {format_time(time.time()-tic)}\n")

    save_checkpoint(
        filename=os.path.join(config.OUTPUT_PATH, f"E_{config.NAME}.pt"),
        model=ensembler.state_dict(),
        optimizer=optimizer.state_dict(),
        epoch=epoch,
    )
    if config.USE_WANDB:
        wandb.save(os.path.join(config.OUTPUT_PATH, f"E_{config.NAME}.pt"))
    torch.cuda.empty_cache()
