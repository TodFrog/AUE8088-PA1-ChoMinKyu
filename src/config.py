import os
from torchvision import models
import torch.nn as nn
import torch

# Training Hyperparameters
NUM_CLASSES         = 200
BATCH_SIZE          = 128
VAL_EVERY_N_EPOCH   = 1


NUM_EPOCHS          = 40
OPTIMIZER_PARAMS    = {'type': 'AdamW', 'lr': 5e-4, 'weight_decay': 0.05}
SCHEDULER_PARAMS    = {'type': 'CosineAnnealingLR', 'T_max': NUM_EPOCHS, 'eta_min': 1e-6}

# Dataaset
DATASET_ROOT_PATH = r"C:\datasets\datasets"
NUM_WORKERS         = 8

# Augmentation
IMAGE_ROTATION      = 20
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 64
IMAGE_PAD_CROPS     = 4
IMAGE_MEAN          = [0.485, 0.456, 0.406]
IMAGE_STD           = [0.229, 0.224, 0.225]

# Network
MODEL_NAME          = 'MyNetwork_alexnet'

# Compute related
ACCELERATOR         = 'gpu'
DEVICES             = [0]
PRECISION_STR       = '16-mixed'
torch.backends.cuda.matmul.allow_tf32 = True

# Logging
WANDB_PROJECT       = 'aue8088-pa1'
WANDB_ENTITY        = os.environ.get('WANDB_ENTITY')
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50
WANDB_NAME          = f'{MODEL_NAME}-B{BATCH_SIZE}-{OPTIMIZER_PARAMS["type"]}'
WANDB_NAME         += f'-{SCHEDULER_PARAMS["type"]}{OPTIMIZER_PARAMS["lr"]:.1E}'
