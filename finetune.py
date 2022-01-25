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
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as A
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

import models
from datasets import CloudDataset
from losses import CloudLoss
from utils import *
from loguru import logger

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
logger.add(config.LOG_FILE)
logger.info(f"Using device: {config.DEVICE}")
seed_everything(config.SEED)
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
    logger.info("Finetuning with {} chips".format(
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

    generator = dataloader.__iter__()
    loss_fn = CloudLoss(
        config.LOSS_CFG
    )
    model = getattr(models, config.MODEL)(
        **config.MODEL_PARAMS[config.MODEL]
    ).to(config.DEVICE)
    model = load_model_weights(model, config.NAME + '.pt', folder=config.OUTPUT_PATH)
    optimizer = getattr(torch.optim, config.FT_OPTIMIZER_CFG.pop('type'))(
        model.parameters(), **config.FT_OPTIMIZER_CFG
    )
    grad_scaler = torch.cuda.amp.GradScaler(enabled=config.AMP)

    logger.info(f"Model has {count_parameters(model)} parameters")

    for epoch in range(config.FT_EPOCHS):
        tic = time.time()

        # Finetuning
        model.train()
        model.zero_grad()
        bar = tqdm(enumerate(generator), total=len(generator))
        metrics = defaultdict(lambda: 0)
        dataset_len = 0

        for step, batch_data in bar:

            images = batch_data['inputs'].to(config.DEVICE)
            labels = batch_data['labels'].to(config.DEVICE)
            
            with torch.cuda.amp.autocast(enabled=config.AMP):
                preds_dict = model(images)
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
                **{name: value / dataset_len 
                            for name, value in metrics.items()},
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

    save_model_weights(model, config.NAME + '_f.pt', folder=config.OUTPUT_PATH)
    if config.USE_WANDB:
        wandb.save(config.LOG_FILE)
        wandb.save(os.path.join(config.OUTPUT_PATH, config.NAME + '_f.pt'))
    torch.cuda.empty_cache()
