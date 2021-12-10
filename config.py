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

# Training
OPTIMIZER = 'Adam' # Should be one of the algorithms in https://pytorch.org/docs/stable/optim.html
TRAIN_BATCH_SIZE = 16
EPOCHS = 6
LEARNING_RATE = 5e-4
AMP = True

# Validating
VAL_BATCH_SIZE = 16

# Outputs
SAVE_MODEL = False
OUTPUT_PATH = 'outputs/'

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)