"""
    [AUE8088] PA1: Image Classification
        - To run: (aue8088) $ python train.py
        - For better flexibility, consider using LightningCLI in PyTorch Lightning
"""
# PyTorch & Pytorch Lightning
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning import Trainer
import torch

# Custom packages
from src.dataset import TinyImageNetDatasetModule
from src.network import SimpleClassifier
from src.metric import count_parameters, count_flops

import src.config as cfg

torch.set_float32_matmul_precision('medium')


if __name__ == "__main__":

    model = SimpleClassifier(
        model_name = "MyNetwork",
        num_classes = cfg.NUM_CLASSES,
        optimizer_params = cfg.OPTIMIZER_PARAMS,
        scheduler_params = cfg.SCHEDULER_PARAMS,
    )

    datamodule = TinyImageNetDatasetModule(
        batch_size = cfg.BATCH_SIZE,
    )

    wandb_logger = WandbLogger(
        project = cfg.WANDB_PROJECT,
        save_dir = cfg.WANDB_SAVE_DIR,
        entity = cfg.WANDB_ENTITY,
        name = cfg.WANDB_NAME,
    )

    trainer = Trainer(
        accelerator = cfg.ACCELERATOR,
        devices = cfg.DEVICES,
        precision = cfg.PRECISION_STR,
        max_epochs = cfg.NUM_EPOCHS,
        check_val_every_n_epoch = cfg.VAL_EVERY_N_EPOCH,
        logger = wandb_logger,
        callbacks = [
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(save_top_k=1, monitor='accuracy/val', mode='max'),
        ],
    )

    trainer.fit(model, datamodule=datamodule)
    val_results = trainer.validate(ckpt_path='best', datamodule=datamodule)

    # Count Params / FLOPS
    num_params = count_parameters(model)
    dummy_input = torch.randn(1, 3, 64, 64).to(model.device)
    flops = count_flops(model, dummy_input)

    # Log to WandB
    top1_acc = val_results[0]['accuracy/val']
    wandb_logger.log_metrics({
        "num_parameters": num_params,
        "flops": flops,
        "top1_accuracy_final": top1_acc,
    })
    
