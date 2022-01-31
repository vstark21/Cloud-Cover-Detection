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
from losses import CloudLoss
from utils import *
from loguru import logger

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', dest='config', type=str, help='Path to the config file', default='config.yml')
parser.add_argument('--resume_id', dest='resume_id', type=str, help='Resume training', default=None)
args = parser.parse_args()

# config
with open(args.config, "r") as f:
    config = AttrDict(yaml.safe_load(f))
if config.DEBUG:
    config.USE_WANDB = False
    config.AMP = False
    config.N_ACCUMULATE = 2

config.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
config.CHECKPOINT_PATH = os.path.join(config.OUTPUT_PATH, config.NAME + '.pt')
if not os.path.exists(config.OUTPUT_PATH):
    os.makedirs(config.OUTPUT_PATH)
logger.info(f"Using device: {config.DEVICE}")
seed_everything(config.SEED)
torch.autograd.set_detect_anomaly(config.DEBUG)

if __name__ == "__main__":
    # Loading data
    files = []
    bad_chips = json.load(
        open(config.BAD_CHIPS_FILE, "r")
    )
    for name in glob.glob(
        os.path.join(config.DATA_PATH, "*.npz")
    ):
        chip_id = os.path.split(name)[-1].split(".")[0]
        if chip_id in bad_chips or 'label' in chip_id:
            continue    
        files.append({
            "chip_id": chip_id,
            "feat_path": name,
            "label_path": name.replace(".npz", "_label.npz")
        })
    train_files, val_files = train_test_split(files, test_size=0.2, 
                                            random_state=config.SEED)
    logger.info("Training with {} chips and Validating with {} chips".format(
        len(train_files), len(val_files)
    ))
    train_transform = A.Compose([
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.5),
    ])
    train_dataset = CloudDataset(
        train_files, config.DATA_MEAN, config.DATA_STD, use_bands=config.USE_BANDS, transforms=train_transform)
    val_dataset = CloudDataset(
        val_files, config.DATA_MEAN, config.DATA_STD, use_bands=config.USE_BANDS)
    
    train_dataloader = DataLoader(
                        train_dataset,
                        batch_size=config.TRAIN_BATCH_SIZE,
                        shuffle=True,
                        num_workers=config.NUM_WORKERS,
                        worker_init_fn=worker_init_fn)
    val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=config.VAL_BATCH_SIZE,
                    shuffle=False,
                    num_workers=config.NUM_WORKERS)
    train_generator = train_dataloader.__iter__()

    loss_fn = CloudLoss(
        config.LOSS_CFG
    )
    model = getattr(models, config.MODEL)(
        **config.MODEL_PARAMS[config.MODEL]
    ).to(config.DEVICE)
    optimizer = getattr(torch.optim, config.OPTIMIZER_CFG.pop('type'))(
        model.parameters(), **config.OPTIMIZER_CFG
    )
    scheduler = getattr(torch.optim.lr_scheduler, config.SCHEDULER)(
        optimizer, **config.SCHEDULER_PARAMS
    )
    grad_scaler = torch.cuda.amp.GradScaler(enabled=config.AMP)
    init_epoch = 0
    best_val_js = 0
    if config.USE_WANDB:
        import wandb
        if not args.resume_id:
            wandb.init(
                project=config.NAME, 
                entity="vstark21", 
                config=config,
            )
        else:
            wandb.init(
                project=config.NAME, 
                entity="vstark21",
                id=args.resume_id,
                resume='must'
            )
            checkpoint = load_checkpoint(wandb.restore(config.CHECKPOINT_PATH).name)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            init_epoch = checkpoint['epoch'] + 1
            best_val_js = checkpoint['best_val']
            logger.info(f"Resuming training from epoch {init_epoch}")

    logger.info(f"Model has {count_parameters(model)} parameters")

    for epoch in range(init_epoch, config.EPOCHS):
        tic = time.time()

        # Training
        model.train()
        model.zero_grad()
        bar = tqdm(range(config.TRAIN_ITERS), total=config.TRAIN_ITERS)
        train_metrics = defaultdict(lambda: 0)
        dataset_len = 0

        for step in bar:
            try:
                batch_data = next(train_generator)
            except StopIteration:
                train_generator = train_dataloader.__iter__()
                batch_data = next(train_generator)

            images = batch_data['inputs'].to(config.DEVICE)
            labels = batch_data['labels'].to(config.DEVICE)
            
            with torch.cuda.amp.autocast(enabled=config.AMP):
                preds_dict = model(images)
                loss_dict = loss_fn(preds_dict, labels)
                loss = loss_dict['loss']

                for name, value in loss_dict.items():
                    train_metrics[name] += value.item()

                train_metrics['jaccard'] += jaccard_score(preds_dict['out'], labels).item()

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
                            for name, value in train_metrics.items()},
                lr=optimizer.param_groups[0]['lr']
            )
        train_metrics = {
            name: value / dataset_len for name, value in train_metrics.items()
        }
        scheduler.step(train_metrics['loss'])

        # Validation
        with torch.no_grad():
            model.eval()
            bar = tqdm(val_dataloader, total=len(val_dataloader))
            val_metrics = defaultdict(lambda: 0)
            dataset_len = 0

            for batch_data in bar:

                images = batch_data['inputs'].to(config.DEVICE)
                labels = batch_data['labels'].to(config.DEVICE)

                preds_dict = model(images)
                loss_dict = loss_fn(preds_dict, labels)
                loss = loss_dict['loss']

                for name, value in loss_dict.items():
                    val_metrics[name] += value.item()

                val_metrics['jaccard'] += jaccard_score(preds_dict['out'], labels).item()
                dataset_len += 1

                bar.set_postfix(
                    epoch=epoch, 
                    **{name: value / dataset_len 
                                for name, value in val_metrics.items()},
                )

        val_metrics = {
            name: value / dataset_len for name, value in val_metrics.items()
        }
        
        for name, value in train_metrics.items():
            logger.info(f"train {name}: {value}")
        for name, value in val_metrics.items():
            logger.info(f"val {name}: {value}")

        if config.USE_WANDB:
            log_dict = dict()
            for name, value in train_metrics.items():
                log_dict[f"train_{name}"] = value
            for name, value in val_metrics.items():
                log_dict[f"val_{name}"] = value
            log_dict['lr'] = optimizer.param_groups[0]['lr']
            wandb.log(log_dict)

        if best_val_js < val_metrics['jaccard']:
            best_val_js = val_metrics['jaccard']
            save_checkpoint(
                filename=config.CHECKPOINT_PATH,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                epoch=epoch,
                best_val=best_val_js,
            )
            if config.USE_WANDB:
                wandb.save(config.CHECKPOINT_PATH)
        logger.info(f"Epoch {epoch} ended, time taken {format_time(time.time()-tic)}\n")

        if optimizer.param_groups[0]['lr'] <= config.MIN_LEARNING_RATE:
            logger.info(f"Learning rate has reached its minimum value, stopping training at {epoch}")
            break

    torch.cuda.empty_cache()
