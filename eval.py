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

import torch
from torch.utils.data import DataLoader

import models
from datasets import CloudDataset
from utils import *
from loguru import logger

# config
with open("config.yml", "r") as f:
    config = AttrDict(yaml.safe_load(f))

config.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {config.DEVICE}")

if __name__ == "__main__":
    files = []
    bad_chips = json.load(
        open(os.path.join(config.DATA_PATH, config.BAD_CHIPS_FILE), "r")
    )
    meta_data = pd.read_csv(
        os.path.join(config.DATA_PATH, config.META_DATA_FILE)
    )
    locations = config.LOCATIONS
    for name in glob.glob(os.path.join(config.DATA_PATH, "*.npz")):
        chip_id = name[-8:-4]
        cur_loc = meta_data.loc[meta_data['chip_id'] == chip_id, 'location'].values[0]
        cur_dt = meta_data.loc[meta_data['chip_id'] == chip_id, 'datetime'].values[0]
        cur_dt = datetime.datetime.strptime(cur_dt, "%Y-%m-%dT%H:%M:%SZ")
        meta = np.zeros(len(locations) + 12)
        meta[locations.index(cur_loc)] = 1
        meta[len(locations) + cur_dt.month - 1] = 1        
        files.append({
            "chip_id": chip_id,
            "path": name,
            "meta": meta
        })
    val_dataset = CloudDataset(
        files, config.DATA_MEAN, config.DATA_STD)
    dataloader = DataLoader(
                    val_dataset,
                    batch_size=config.VAL_BATCH_SIZE,
                    shuffle=False,
                    num_workers=config.NUM_WORKERS)

    model = getattr(models, config.MODEL)(
        **config.MODEL_PARAMS[config.MODEL]
    )
    model = load_model_weights(model, config.NAME + '.pt', folder=config.OUTPUT_PATH)
    model = model.to(config.DEVICE)

    logger.info(f"Model has {count_parameters(model)} parameters")

    with torch.no_grad():
        model.eval()
        bar = tqdm(dataloader, total=len(dataloader))
        jac_score = 0
        dataset_len = 0

        for batch_data in bar:

            images = batch_data['inputs'].to(config.DEVICE)
            meta = batch_data['meta'].to(config.DEVICE)
            labels = batch_data['labels'].to(config.DEVICE)

            preds_dict = model(images, meta)
            cur_jac = jaccard_score(preds_dict['out'], labels).item()
            jac_score += cur_jac
            dataset_len += 1

            bar.set_postfix(
                jac_score=jac_score / dataset_len,
            )
    logger.info(f"Jaccard Score on entire data: {jac_score / dataset_len}")

    torch.cuda.empty_cache()
