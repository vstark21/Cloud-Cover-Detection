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
from trainer import Trainer
from utils import *
from loguru import logger


paths = [
    "/content/westeurope"
]

files = []
for name in paths:
    _files = [os.path.join(name, el) for el in os.listdir(name)]
    files.extend(_files)

train_files, val_files = train_test_split(files, test_size=0.25, 
                                          random_state=42)

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

loss_fn = CloudLoss()
model = CloudModel(n_channels=config.N_CHANNELS,
                   n_classes=config.N_CLASSES).to(config.DEVICE)
optimizer = getattr(torch.optim, config.OPTIMIZER)()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
grad_scaler = torch.cuda.amp.GradScaler(enabled=config.AMP)
trainer = Trainer(model, optimizer, loss_fn, scheduler)

best_val_loss = float('inf')

for epoch in range(config.EPOCHS):
    tic = time.time()

    train_loss = trainer.train(train_dataloader, epoch, grad_scaler)
    val_loss = trainer.evaluate(val_dataloader, epoch)

    if best_val_loss > val_loss:
        best_val_loss = val_loss
        save_model_weights(model, config.NAME + '.pt', folder=config.OUTPUT_PATH)

    logger.info('\n', '-'*15, f" Epoch {epoch} ended, time taken {format_time(time.time()-tic)} ", '-'*15)

torch.cuda.empty_cache()
