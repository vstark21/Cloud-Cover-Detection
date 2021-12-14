# General imports
import os
import cv2
import time
import json
import scipy
import wandb
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

import config
from models import Unet, CloudNetp
from datasets import CloudDataset
from losses import CloudLoss
from utils import *
from loguru import logger

files = []
for name in config.DATA_PATHS:
    _files = [os.path.join(name, el) for el in os.listdir(name)]
    files.extend(_files)

train_files, val_files = train_test_split(files, test_size=0.25, 
                                          random_state=config.SEED)

train_dataset = CloudDataset(train_files)
val_dataset = CloudDataset(val_files)

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
val_generator = val_dataloader.__iter__()
loss_fn = CloudLoss(
    bce_lw=config.BCE_LW,
    dice_lw=config.DICE_LW,
    jacc_lw=config.JACC_LW
)
if config.MODEL == 'unet':
    model = Unet(
        n_channels=config.N_CHANNELS,
        n_classes=config.N_CLASSES,
        model_size=config.MODEL_SIZE)
elif config.MODEL == 'cloudnetp':
    model = CloudNetp(
        n_channels=config.N_CHANNELS,
        n_classes=config.N_CLASSES,
        inception_depth=config.INCEPTION_DEPTH,
        model_size=config.MODEL_SIZE)

model = model.to(config.DEVICE)
optimizer = getattr(torch.optim, config.OPTIMIZER)(model.parameters(), lr=config.LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
grad_scaler = torch.cuda.amp.GradScaler(enabled=config.AMP)

logger.info(f"Model has {count_parameters(model)} parameters")

best_val_js = 0

for epoch in range(config.EPOCHS):
    tic = time.time()

    # Training
    model.train()
    model.zero_grad()
    bar = tqdm(range(config.TRAIN_ITERS), total=config.TRAIN_ITERS)
    train_loss, train_bce_loss, train_dice_loss = 0, 0, 0
    dataset_len = 0
    train_jacc_score = 0

    for i in bar:
        try:
            batch_data = next(train_generator)
        except StopIteration:
            train_generator = train_dataloader.__iter__()
            batch_data = next(train_generator)

        images = batch_data['inputs'].to(config.DEVICE)
        labels = batch_data['labels'].to(config.DEVICE)
        
        with torch.cuda.amp.autocast(enabled=config.AMP):
            preds = model(images)
            loss, bce_loss, dice_loss = loss_fn(preds, labels)
            train_jacc_score += jaccard_score(preds, labels).item()

        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

        train_loss += loss.item()
        train_bce_loss += bce_loss.item()
        train_dice_loss += dice_loss.item()
        dataset_len += 1

        bar.set_postfix(
            epoch=epoch, 
            total_loss=train_loss / dataset_len,
            bce_loss=train_bce_loss / dataset_len,
            dice_loss=train_dice_loss / dataset_len,
            jacc_score=train_jacc_score / dataset_len,
            lr=optimizer.param_groups[0]['lr']
        )
    train_loss /= dataset_len
    train_bce_loss /= dataset_len
    train_dice_loss /= dataset_len
    train_jacc_score /= dataset_len
    scheduler.step(train_loss)

    # Validation
    with torch.no_grad():
        model.eval()
        bar = tqdm(range(config.VAL_ITERS), total=config.VAL_ITERS)
        val_loss, val_bce_loss, val_dice_loss = 0, 0, 0
        dataset_len = 0
        val_jacc_score = 0

        for i in bar:
            try:
                batch_data = next(val_generator)
            except StopIteration:
                val_generator = val_dataloader.__iter__()
                batch_data = next(val_generator)

            images = batch_data['inputs'].to(config.DEVICE)
            labels = batch_data['labels'].to(config.DEVICE)

            preds = model(images)
            loss, bce_loss, dice_loss = loss_fn(preds, labels)

            val_loss += loss.item()
            val_bce_loss += bce_loss.item()
            val_dice_loss += dice_loss.item()
            val_jacc_score += jaccard_score(preds, labels).item()
            dataset_len += 1

            bar.set_postfix(
                epoch=epoch, 
                total_loss=val_loss / dataset_len,
                bce_loss=val_bce_loss / dataset_len,
                dice_loss=val_dice_loss / dataset_len,
                jacc_score=val_jacc_score / dataset_len
            )

    val_loss /= dataset_len
    val_bce_loss /= dataset_len
    val_dice_loss /= dataset_len
    val_jacc_score /= dataset_len

    logger.info(f"train loss: {train_loss}")
    logger.info(f"train bce loss: {train_bce_loss}")
    logger.info(f"train dice loss: {train_dice_loss}")
    logger.info(f"train jaccard: {train_jacc_score}")
    logger.info(f"val loss: {val_loss}")
    logger.info(f"val bce loss: {val_bce_loss}")
    logger.info(f"val dice loss: {val_dice_loss}")
    logger.info(f"val jaccard: {val_jacc_score}")

    if config.USE_WANDB:
        wandb.log({
            "train_loss": train_loss,
            "train_bce_loss": train_bce_loss,
            "train_dice_loss": train_dice_loss,
            "train_jaccard": train_jacc_score,
            "val_loss": val_loss,
            "val_bce_loss": val_bce_loss,
            "val_dice_loss": val_dice_loss,
            "val_jaccard": val_jacc_score,
            "lr": optimizer.param_groups[0]['lr']
        })

    if best_val_js < val_jacc_score:
        best_val_js = val_jacc_score
        save_model_weights(model, config.NAME + '.pt', folder=config.OUTPUT_PATH)

    logger.info(f"Epoch {epoch} ended, time taken {format_time(time.time()-tic)}\n")

if config.USE_WANDB:
    wandb.save(os.path.join(config.OUTPUT_PATH, config.NAME + '.pt'))
    wandb.save(config.LOG_FILE)

torch.cuda.empty_cache()
