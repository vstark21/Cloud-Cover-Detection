# General imports
import os
import cv2
import glob
import time
import json
import scipy
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

import config
from model import CloudModel
from dataset import CloudDataset
from loss import CloudLoss
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
loss_fn = CloudLoss()
model = CloudModel(n_channels=config.N_CHANNELS,
                   n_classes=config.N_CLASSES).to(config.DEVICE)
optimizer = getattr(torch.optim, config.OPTIMIZER)(model.parameters(), lr=config.LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
grad_scaler = torch.cuda.amp.GradScaler(enabled=config.AMP)

logger.info(f"Model has {count_parameters(model)} parameters")

best_val_loss = float('inf')

for epoch in range(config.EPOCHS):
    tic = time.time()

    # Training
    model.train()
    model.zero_grad()
    bar = tqdm(range(config.TRAIN_ITERS), total=config.TRAIN_ITERS)
    train_epoch_loss = 0
    dataset_len = 0
    jacc_score = 0

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
            loss = loss_fn(preds, labels)
            jacc_score += jaccard_score(preds, labels).item()

        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

        train_epoch_loss += loss.item()
        dataset_len += 1

        bar.set_postfix(epoch=epoch, loss=train_epoch_loss / dataset_len,
                        jaccard=jacc_score / dataset_len,
                    lr=optimizer.param_groups[0]['lr'])
    
    scheduler.step(train_epoch_loss / dataset_len)

    # Validation
    with torch.no_grad():
        model.eval()
        bar = tqdm(range(config.VAL_ITERS), total=config.VAL_ITERS)
        val_epoch_loss = 0
        dataset_len = 0
        jacc_score = 0

        for i in bar:
            try:
                batch_data = next(val_generator)
            except StopIteration:
                val_generator = val_dataloader.__iter__()
                batch_data = next(val_generator)

            images = batch_data['inputs'].to(config.DEVICE)
            labels = batch_data['labels'].to(config.DEVICE)

            preds = model(images)
            loss = loss_fn(preds, labels)
            jacc_score += jaccard_score(preds, labels).item()

            val_epoch_loss += loss.item()
            dataset_len += 1

            bar.set_postfix(epoch=epoch, loss=val_epoch_loss / dataset_len,
                        jaccard=jacc_score / dataset_len)

    if best_val_loss > val_epoch_loss:
        best_val_loss = val_epoch_loss
        save_model_weights(model, config.NAME + '.pt', folder=config.OUTPUT_PATH)

    logger.info(f"Epoch {epoch} ended, time taken {format_time(time.time()-tic)}\n")

torch.cuda.empty_cache()
