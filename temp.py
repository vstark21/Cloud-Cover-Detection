import torch
from models.segmenter.factory import create_segmenter
from utils import *
import yaml
with open("config.yml", "r") as f:
    config = AttrDict(yaml.safe_load(f))

model = create_segmenter(
    **config.MODEL_PARAMS[config.MODEL]
)
a = torch.randn((2, 4, 512, 512))
out = model(a)