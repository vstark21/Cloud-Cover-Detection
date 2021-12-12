import os
import torch
from utils import *
import wandb
from loguru import logger

SEED = 42
VERBOSE = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NAME = 'CCD'

NUM_WORKERS = 2
N_CHANNELS = 4
N_CLASSES = 1

# Data
DATA_PATHS = [
    "data/"
]

# Loss
BCE_LW = 1.0
DICE_LW = 1.0

# Training
OPTIMIZER = 'Adam' # Should be one of the algorithms in https://pytorch.org/docs/stable/optim.html
TRAIN_BATCH_SIZE = 4 
EPOCHS = 50
LEARNING_RATE = 5e-4
AMP = True
TRAIN_ITERS = 1000

# Validating
VAL_BATCH_SIZE = 4
VAL_ITERS = 300

# Wandb 
USE_WANDB = True

# Outputs
OUTPUT_PATH = 'outputs/'
LOG_FILE = 'outputs/logs.log'

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Removing it because loguru.logger will create it
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)
logger.add(LOG_FILE)
logger.info(f"Using device: {DEVICE}")

seed_everything(SEED)

if USE_WANDB:
    wandb.init(
        project=NAME, 
        entity="vstark21", 
        config = {
            "seed": SEED,
            "device": DEVICE,
            "optimizer": OPTIMIZER,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "amp": AMP,
            "train_iters": TRAIN_ITERS,
            "val_batch_size": VAL_BATCH_SIZE,
            "val_iters": VAL_ITERS
        }
    )
