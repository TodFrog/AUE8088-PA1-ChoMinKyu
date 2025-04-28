"""
    [AUE8088] PA1: Image Classification
        - To run: (aue8088) $ python test.py --ckpt_file wandb/aue8088-pa1/ygeiua2t/checkpoints/epoch\=19-step\=62500.ckpt
"""
# Python packages
import argparse

# PyTorch & Pytorch Lightning
from lightning import Trainer
from torch.utils.flop_counter import FlopCounterMode
import torch

# Wandb
import wandb
# Wandb settings
from lightning.pytorch.loggers import WandbLogger


# Custom packages
from src.dataset import TinyImageNetDatasetModule
from src.network import SimpleClassifier
import src.config as cfg

torch.set_float32_matmul_precision('medium')

wandb_logger = WandbLogger(
    project = cfg.WANDB_PROJECT,
    save_dir = cfg.WANDB_SAVE_DIR,
    entity = cfg.WANDB_ENTITY,
    name = cfg.WANDB_NAME + "-test",  # 테스트용으로 이름 뒤에 태그 추가
)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--ckpt_file',
        type = str,
        help = 'Model checkpoint file name')
    args = args.parse_args()

    model = SimpleClassifier(
        model_name = cfg.MODEL_NAME,
        num_classes = cfg.NUM_CLASSES,
    )

    datamodule = TinyImageNetDatasetModule(
        batch_size = 1,
    )

    trainer = Trainer(
        accelerator = cfg.ACCELERATOR,
        devices = cfg.DEVICES,
        precision = cfg.PRECISION_STR,
        benchmark = True,
        inference_mode = True,
        logger = wandb_logger,
    )

    trainer.validate(model, ckpt_path = args.ckpt_file, datamodule = datamodule)

    # FLOP counter
    x, y = next(iter(datamodule.test_dataloader()))
    x = x.to(model.device)  # 모델이 올라간 디바이스로 tensor도 옮김

    flop_counter = FlopCounterMode(model, depth=1)

    with flop_counter:
        model(x)
