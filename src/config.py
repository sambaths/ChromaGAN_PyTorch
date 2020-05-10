import os
import torch

ROOT_DIR = os.path.abspath('')
DATA_DIR = os.path.join(ROOT_DIR, 'input/')
OUT_DIR = os.path.join(ROOT_DIR, 'result/')
MODEL_DIR = os.path.join(ROOT_DIR, 'models/')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoint/')

TRAIN_DIR = "train/"  # UPDATE
TEST_DIR = "test/" # UPDATE

os.makedirs(DATA_DIR+TRAIN_DIR, exist_ok=True)
os.makedirs(DATA_DIR+TEST_DIR, exist_ok=True)

# DATA INFORMATION
IMAGE_SIZE = 224
BATCH_SIZE = 16
GRADIENT_PENALTY_WEIGHT = 10
NUM_EPOCHS = 10
KEEP_CKPT = 1
# save_model_path = MODEL_DIR

if torch.cuda.is_available():
  DEVICE = torch.device('cuda')
else:
  DEVICE = 'cpu'
