from loguru import logger
import numpy as np
import cv2
import torch
import models
from PIL import Image
import typer
import os
import time
import yaml
import datetime
import pandas as pd

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

with open("config.yml", "r") as f:
    config = AttrDict(yaml.safe_load(f))
config.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ROOT_DIRECTORY = "/codeexecution"
if not os.path.exists(ROOT_DIRECTORY):
    ROOT_DIRECTORY = "testexecution"
PREDICTIONS_DIRECTORY = os.path.join(ROOT_DIRECTORY, "predictions")
feature_directory = os.path.join(ROOT_DIRECTORY, "data", "test_features")

chips = os.listdir(feature_directory)
logger.info(f"Processing {len(chips)} chips in {feature_directory}")
model = getattr(models, config.MODEL)(
    **config.MODEL_PARAMS[config.MODEL]
).to(config.DEVICE)
model.load_state_dict(torch.load(F"{config.NAME}.pt", map_location=config.DEVICE))
model.eval()

@torch.no_grad()
def main():
    tic = time.time()
    for i, chip in enumerate(chips):
        files = [os.path.join(feature_directory, chip, file) 
                    for file in ["B02.tif", "B03.tif", "B04.tif", "B08.tif"]]
        inputs = []
        for filename in files:
            _band = np.array(Image.open(filename))
            key = os.path.split(filename)[-1].split(".")[0]
            _band = (_band - config.DATA_MEAN[key]) / config.DATA_STD[key]
            inputs.append(_band)

        inputs = np.stack(inputs, axis=0)
        inputs = inputs.astype(np.float32)
        inputs = inputs[np.newaxis, :, :, :]
        inputs = torch.Tensor(inputs)
        inputs = inputs.to(config.DEVICE)

        pred = model(inputs)['out']
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()
        pred = np.round(pred).astype(np.uint8)

        Image.fromarray(pred).save(
            os.path.join(PREDICTIONS_DIRECTORY, f"{chip}.tif")
        )
        if i % 100 == 0:
            logger.info(f"{100 * i / len(chips)}% processed")
    logger.info(f"Processed {len(chips)} chips in {time.time() - tic} seconds")


if __name__ == "__main__":
    typer.run(main)