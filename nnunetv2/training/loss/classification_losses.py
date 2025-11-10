import torch
from torch import nn


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, class_counts, beta: float = 0.9999, gamma: float = 2.0, eps: float = 1e-8):
        super().__init__()
        n = torch.as_tensor(class_counts, dtype=torch.float)
        eff = (1.0 - beta) / (1.0 - torch.clamp(beta ** n, min=1e-12))
        weights = eff * (len(n) / torch.clamp(eff.sum(), min=eps))
        self.register_buffer('cls_weights', weights)
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logpt = torch.log_softmax(logits, dim=1)
        pt = torch.exp(logpt)
        idx = torch.arange(target.size(0), device=logits.device)
        gt_logpt = logpt[idx, target]
        gt_pt = pt[idx, target]
        weights = self.cls_weights.to(logits.device)
        at = weights[target]
        loss = -at * ((1.0 - gt_pt).clamp(min=self.eps) ** self.gamma) * gt_logpt
        return loss.mean()


class LDAMLoss(nn.Module):
    def __init__(self, class_counts, max_m: float = 0.5, s: int = 30, drw_weight=None):
        super().__init__()
        n = torch.as_tensor(class_counts, dtype=torch.float)
        margins = max_m / (n.clamp(min=1.0).pow(0.25))
        self.register_buffer('margins', margins)
        if drw_weight is not None:
            w = torch.as_tensor(drw_weight, dtype=torch.float)
            self.register_buffer('drw_weight', w / w.mean())
        else:
            self.drw_weight = None
        self.s = s

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        margins = self.margins.to(logits.device)
        one_hot = torch.zeros_like(logits).scatter_(1, target.view(-1, 1), 1)
        logits_m = logits - one_hot * margins[target].unsqueeze(1)
        logits_m = self.s * logits_m
        loss = nn.functional.cross_entropy(logits_m, target, reduction='none')
        if self.drw_weight is not None:
            w = self.drw_weight.to(logits.device)[target]
            loss = loss * w
        return loss.mean()

