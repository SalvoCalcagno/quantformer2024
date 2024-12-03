from src.metrics.FocalLoss import FocalLoss
from torch.nn import MSELoss
import torch.nn as nn

class MaskedMSELoss(nn.Module):
    
        def __init__(self):
            super(MaskedMSELoss, self).__init__()
    
        def forward(self, preds, target, mask):
            loss = (preds - target) ** 2
            loss = loss.mean(dim=-1)
            loss = (loss * mask).sum() / mask.sum()
            return loss

class PretrainLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(PretrainLoss, self).__init__()
        self.focal_loss = FocalLoss(
            alpha=alpha, gamma=gamma, reduction=reduction
        )
        self.mse_loss = MSELoss()
        self.masked_mse_loss = MaskedMSELoss()

    def forward(self, pred, target, mask=None):
        pred_cls, pred_reg = pred
        target_cls, target_reg = target

        cls_loss = self.focal_loss(pred_cls, target_cls)
        if mask is not None:
            reg_loss = self.masked_mse_loss(pred_reg, target_reg, mask)
        else:
            reg_loss = self.mse_loss(pred_reg, target_reg)
        #print(cls_loss, reg_loss)

        return 2*cls_loss + reg_loss