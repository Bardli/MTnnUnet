import torch
from torch import nn
from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.ddp_allgather import AllGatherGrad


class MemoryEfficientTverskyLoss(nn.Module):
    def __init__(
        self,
        apply_nonlin=None,
        batch_dice: bool = False,
        do_bg: bool = True,
        smooth: float = 1e-5,
        ddp: bool = True,
        alpha: float = 0.3,
        beta: float = 0.7,
        focal_gamma: float = None,
    ):
        super().__init__()
        self.apply_nonlin = apply_nonlin if apply_nonlin is not None else softmax_helper_dim1
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.ddp = ddp
        self.alpha = alpha
        self.beta = beta
        self.focal_gamma = focal_gamma

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        axes = tuple(range(2, x.ndim))
        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, square=False)

        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        if not self.do_bg:
            if self.batch_dice:
                tp, fp, fn = tp[1:], fp[1:], fn[1:]
            else:
                tp, fp, fn = tp[:, 1:], fp[:, 1:], fn[:, 1:]

        denom = tp + self.alpha * fp + self.beta * fn
        ti = (tp + self.smooth) / torch.clamp(denom + self.smooth, min=1e-8)
        loss = 1.0 - ti

        if self.focal_gamma is not None and self.focal_gamma > 1.0:
            loss = loss ** self.focal_gamma

        return loss.mean()

