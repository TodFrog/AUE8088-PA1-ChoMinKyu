import os
import torch

# Training Hyperparameters
NUM_CLASSES         = 200
BATCH_SIZE          = 256
VAL_EVERY_N_EPOCH   = 1


NUM_EPOCHS          = 40
OPTIMIZER_PARAMS    = {'type': 'SGD', 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 1e-4}
SCHEDULER_PARAMS    = {'type': 'CosineAnnealingLR', 'T_max': NUM_EPOCHS, 'eta_min': 1e-6}

# Dataaset
DATASET_ROOT_PATH = r"C:\datasets\datasets"
NUM_WORKERS         = 8

# Augmentation
IMAGE_ROTATION      = 45
IMAGE_FLIP_PROB     = 0.7
IMAGE_NUM_CROPS     = 64
IMAGE_PAD_CROPS     = 4
IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
IMAGE_STD           = [0.2302, 0.2265, 0.2262]

# Network
MODEL_NAME          = 'efficientnet_b7'

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
