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

import torch
from torch.utils.data import DataLoader

import models
from datasets import CloudDataset
from utils import *
from loguru import logger

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_ids', dest='model_ids', type=str, help='Model ids', default=None)
parser.add_argument('--ensembler_id', dest='ensembler_id', type=str, help='Ensembler id', default=None)
args = parser.parse_args()

# config
with open("config.yml", "r") as f:
    config = AttrDict(yaml.safe_load(f))

config.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {config.DEVICE}")

if __name__ == "__main__":
    files = []
    bad_chips = json.load(
        open(config.BAD_CHIPS_FILE, "r")
    )
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
    val_dataset = CloudDataset(
        files, config.DATA_MEAN, config.DATA_STD, use_bands=config.USE_BANDS)
    dataloader = DataLoader(
                    val_dataset,
                    batch_size=config.VAL_BATCH_SIZE,
                    shuffle=False,
                    num_workers=config.NUM_WORKERS)

    ensemble_models = []
    if args.model_ids is None:
        raise ValueError("Please provide model ids")
    elif args.ensembler_id is None:
        raise ValueError("Please provide ensembler id")
    else:
        logger.info(f"Evaluating {args.model_ids} with {args.ensembler_id}")
        model_ids = args.model_ids.split(",")
        for model_id in model_ids:
            _model = getattr(models, config.FT_MODELS[model_id]['MODEL'])(
                **config.FT_MODELS[model_id]['MODEL_PARAMS']
            ).to(config.DEVICE)
            checkpoint = torch.load(os.path.join(config.OUTPUT_PATH, model_id + '.pt'))
            if 'model' in checkpoint.keys():
                checkpoint = checkpoint['model']
            _model.load_state_dict(checkpoint)
            _model.eval()
            ensemble_models.append(_model)
        ensembler = getattr(models, config.FT_ENSEMBLER)(
            in_channels=len(ensemble_models), out_channels=1
        ).to(config.DEVICE)
        checkpoint = torch.load(os.path.join(config.OUTPUT_PATH, f"E_{args.ensembler_id}.pt"))
        if 'model' in checkpoint.keys():
            checkpoint = checkpoint['model']
        ensembler.load_state_dict(checkpoint)
        ensembler.eval()

    with torch.no_grad():
        bar = tqdm(dataloader, total=len(dataloader))
        jac_score = 0
        dataset_len = 0

        for batch_data in bar:

            images = batch_data['inputs'].to(config.DEVICE)
            labels = batch_data['labels'].to(config.DEVICE)

            images = torch.cat([
                _model(images)['out'] for _model in ensemble_models
            ], dim=1)
            preds_dict = ensembler(images)
            cur_jac = jaccard_score(preds_dict['out'], labels).item()
            jac_score += cur_jac
            dataset_len += 1

            bar.set_postfix(
                jac_score=jac_score / dataset_len,
            )
    logger.info(f"Jaccard Score on entire data: {jac_score / dataset_len}")

    torch.cuda.empty_cache()
