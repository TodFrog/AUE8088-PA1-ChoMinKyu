from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    """ 
    Multi-class F1 score (one-vs-rest) from scratch.
    - preds: (B, C) logits (or probabilities)
    - target: (B,) integer class labels in [0..num_classes-1]
    """

    def __init__(self, num_classes: int, average: bool = True, dist_sync_on_step=False):
        """
        Args:
            num_classes (int): 총 클래스 개수
            average (bool): True 시, 전체 클래스에 대한 macro-average F1 스코어를 반환.
                            False 시, 각 클래스별 F1을 담은 (num_classes,) 텐서를 반환.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.average = average
        
        # 각 클래스별 TP, FP, FN을 저장할 state
        # dist_reduce_fx='sum' -> DDP(멀티 프로세스) 환경에서 텐서들이 'sum'으로 집계됨
        self.add_state("tp", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        preds: (B, C) tensor (로짓 또는 확률)
        target: (B,) tensor (정답 라벨)
        """
        # 1) 가장 큰 logit(혹은 확률)을 갖는 클래스 index로 예측 라벨 결정 (B,)
        preds = torch.argmax(preds, dim=1)

        # 2) preds.shape, target.shape 맞춰주기
        if preds.shape != target.shape:
            target = target.view(-1)

        # 3) 클래스별 TP/FP/FN 계산
        #    one-vs-rest 관점:
        #    - TP[c]: 예측 == c 이고 실제 == c
        #    - FP[c]: 예측 == c 이고 실제 != c
        #    - FN[c]: 예측 != c 이고 실제 == c
        for c in range(self.num_classes):
            # 예측 == c
            pred_c_mask = (preds == c)
            # 실제 == c
            target_c_mask = (target == c)

            tp_c = (pred_c_mask & target_c_mask).sum()
            fp_c = (pred_c_mask & ~target_c_mask).sum()
            fn_c = (~pred_c_mask & target_c_mask).sum()

            self.tp[c] += tp_c
            self.fp[c] += fp_c
            self.fn[c] += fn_c

    def compute(self):
        """
        Returns:
            - 만약 self.average=True 일 때: (scalar) 모든 클래스 F1의 macro-average
            - self.average=False 일 때: (num_classes,) shape 각 클래스별 F1
        """
        # precision[c] = TP[c] / (TP[c] + FP[c])
        # recall[c]    = TP[c] / (TP[c] + FN[c])
        # F1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c])  (분모=0이면 0)

        tp = self.tp.float()
        fp = self.fp.float()
        fn = self.fn.float()

        precision = tp / (tp + fp).clamp_min(1e-12)
        recall    = tp / (tp + fn).clamp_min(1e-12)
        f1_per_class = 2 * precision * recall / (precision + recall).clamp_min(1e-12)

        if self.average:
            return f1_per_class.mean()  # macro-average F1
        else:
            return f1_per_class  # 각 클래스별 F1

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        if preds.shape != target.shape:
            target = target.view(-1)

        # [TODO] Cound the number of correct prediction
        correct = (preds == target).sum()

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
