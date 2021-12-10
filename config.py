import os
import torch
from loguru import logger

SEED = 42
VERBOSE = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

NAME = 'model'

NUM_WORKERS = 2
N_CHANNELS = 4
N_CLASSES = 1

# Data
DATA_PATHS = [
    "/content/westeurope",
    "/content/eastasia",
]

# Training
OPTIMIZER = 'Adam' # Should be one of the algorithms in https://pytorch.org/docs/stable/optim.html
TRAIN_BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 5e-4
AMP = True
TRAIN_ITERS = 200

# Validating
VAL_BATCH_SIZE = 16
VAL_ITERS = 75

# Outputs
OUTPUT_PATH = 'outputs/'

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)