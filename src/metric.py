from torchmetrics import Metric
import torch
from fvcore.nn import FlopCountAnalysis

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes: int, average: bool = True, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.average = average

        # 누적 TP / FP / FN (DDP 시 ‘sum’으로 집계)
        self.add_state("tp", default=torch.zeros(num_classes, dtype=torch.long),
                       dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_classes, dtype=torch.long),
                       dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_classes, dtype=torch.long),
                       dist_reduce_fx="sum")

    # ----------------------------------------------------------
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # logits (B,C) → label (B,)
        if preds.ndim > 1:
            preds = torch.argmax(preds, dim=1)

        # shape check
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        for c in range(self.num_classes):
            self.tp[c] += torch.sum((preds == c) & (target == c))
            self.fp[c] += torch.sum((preds == c) & (target != c))
            self.fn[c] += torch.sum((preds != c) & (target == c))

    # ----------------------------------------------------------
    def compute(self) -> torch.Tensor:
        precision = self.tp.float() / (self.tp + self.fp + 1e-12)
        recall    = self.tp.float() / (self.tp + self.fn + 1e-12)
        f1        = 2 * precision * recall / (precision + recall + 1e-12)
        return f1.mean() if self.average else f1

class MyAccuracy(Metric):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('total',   default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        if preds.ndim > 1:
            preds = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        # [TODO] Cound the number of correct prediction
        correct = (preds == target).sum()

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self) -> torch.Tensor:
        return self.correct.float() / (self.total.float() + 1e-12)
    
    
def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_flops(model: torch.nn.Module, tuple = (1, 3, 64, 64)) -> int:
    
    """
    Count the number of FLOPs in a PyTorch model.
    """
    model.eval()                                      # BatchNorm 등 고정
    dummy = torch.randn(1, 3, 64, 64,
                        device=next(model.parameters()).device)
    flops = FlopCountAnalysis(model, dummy)
    return int(flops.total())
