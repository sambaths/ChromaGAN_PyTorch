USE_TPU = False
MULTI_CORE = False
MIXED_PRECISION = False

import os
import torch

DATA_DIR = '../input/'
OUT_DIR = '../result/'
MODEL_DIR = '../models/'
CHECKPOINT_DIR = '../checkpoint/'
LOGS_DIR = '../logs/'

TRAIN_DIR = DATA_DIR+"train/"  # UPDATE
TEST_DIR = DATA_DIR+"test/" # UPDATE

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# DATA INFORMATION
IMAGE_SIZE = 224
BATCH_SIZE = 1
GRADIENT_PENALTY_WEIGHT = 10
NUM_EPOCHS = 10
KEEP_CKPT = 2
# save_model_path = MODEL_DIR

if USE_TPU:
  import torch_xla.core.xla_model as xm
  if not MULTI_CORE:
    DEVICE = xm.xla_device()

if not USE_TPU:
  if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
  else:
    DEVICE = 'cpu'

if DEVICE=='cpu' and MIXED_PRECISION:
  raise ValueError('To use mixed precision you need GPU')