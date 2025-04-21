# --- 변경된 network.py ------------------------------------------------------
from termcolor import colored
from typing import Dict
import copy

# PyTorch & Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
from torchvision import models
from torchvision.models.alexnet import AlexNet
import torch

# ✅ TorchMetrics
from torchmetrics.classification import Accuracy, MulticlassF1Score

# Custom packages
import src.config as cfg
from src.util import show_setting


class MyNetwork(AlexNet):
    def __init__(self):
        super().__init__()
        # [선택] AlexNet feature extractor 수정하려면 여기서 편집


class SimpleClassifier(LightningModule):
    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 200,
        optimizer_params: Dict = dict(),
        scheduler_params: Dict = dict(),
    ):
        super().__init__()

        # ----------------- Model -----------------
        if model_name == "MyNetwork":
            self.model = MyNetwork()
        else:
            models_list = models.list_models()
            assert (
                model_name in models_list
            ), f"Unknown model name: {model_name}. Choose one from {', '.join(models_list)}"
            self.model = models.get_model(model_name, num_classes=num_classes)

        # ----------------- Loss ------------------
        self.loss_fn = nn.CrossEntropyLoss()

        # ----------------- Metrics ----------------
        # 학습 / 검증용으로 두 세트 분리 (epoch 단위 집계)
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc   = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_f1   = MulticlassF1Score(num_classes=num_classes, average="macro")

        # ----------------- Hparams ---------------
        self.save_hyperparameters()

    # -----------------------------------------------------------
    # Lightning 라이프사이클
    # -----------------------------------------------------------
    def on_train_start(self):
        show_setting(cfg)

    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type   = optim_params.pop("type")
        optimizer    = getattr(torch.optim, optim_type)(self.parameters(), **optim_params)

        sched_params = copy.deepcopy(self.hparams.scheduler_params)
        sched_type   = sched_params.pop("type")
        scheduler    = getattr(torch.optim.lr_scheduler, sched_type)(optimizer, **sched_params)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # -----------------------------------------------------------
    # Forward & shared step
    # -----------------------------------------------------------
    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch):
        x, y   = batch
        logits = self.forward(x)
        loss   = self.loss_fn(logits, y)
        preds  = torch.argmax(logits, dim=1)
        return loss, preds, y

    # -----------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------
    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._common_step(batch)

        # TorchMetrics는 update/compute 패턴
        self.train_acc.update(preds, targets)
        self.train_f1.update(preds, targets)

        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        acc = self.train_acc.compute()
        f1  = self.train_f1.compute()

        self.log_dict(
            {"accuracy/train": acc, "f1/train": f1},
            prog_bar=True,
        )

        # 다음 epoch 집계를 위해 초기화
        self.train_acc.reset()
        self.train_f1.reset()

    # -----------------------------------------------------------
    # Validation loop
    # -----------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._common_step(batch)

        self.val_acc.update(preds, targets)
        self.val_f1.update(preds, targets)

        self.log("loss/val", loss, on_step=False, on_epoch=True, prog_bar=False)

        # (선택) 이미지 로깅
        self._wandb_log_image(batch, batch_idx, preds, frequency=cfg.WANDB_IMG_LOG_FREQ)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        f1  = self.val_f1.compute()

        self.log_dict(
            {"accuracy/val": acc, "f1/val": f1},
            prog_bar=True,
        )

        self.val_acc.reset()
        self.val_f1.reset()

    # -----------------------------------------------------------
    # WandB 이미지 로깅 (기존 코드 유지)
    # -----------------------------------------------------------
    def _wandb_log_image(self, batch, batch_idx, preds, frequency=100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(
                    colored(
                        "Please use WandbLogger to log images.",
                        color="blue",
                        attrs=("bold",),
                    )
                )
            return

        if batch_idx % frequency == 0:
            x, y = batch
            self.logger.log_image(
                key=f"pred/val/batch{batch_idx:05d}_sample_0",
                images=[x[0].cpu()],
                caption=[f"GT: {y[0].item()}, Pred: {preds[0].item()}"],
            )
# ---------------------------------------------------------------------------
