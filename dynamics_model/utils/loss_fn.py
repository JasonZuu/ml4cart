import torch.nn as nn
import torch.nn.functional as F
import torch


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.register_buffer("alpha", alpha if isinstance(alpha, torch.Tensor) else None)

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        one_hot = F.one_hot(targets, num_classes=num_classes).float()
        if self.label_smoothing > 0.0:
            eps = self.label_smoothing
            one_hot = (1 - eps) * one_hot + eps / num_classes
        if self.alpha is not None:
            alpha_t = (self.alpha.unsqueeze(0) * one_hot).sum(dim=1)
        else:
            alpha_t = torch.ones_like(targets, dtype=log_probs.dtype)
        p_t = (probs * F.one_hot(targets, num_classes=num_classes).float()).sum(dim=1)
        focal_factor = (1 - p_t) ** self.gamma
        ce_term = (log_probs * one_hot).sum(dim=1)
        loss = -alpha_t * focal_factor * ce_term
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss