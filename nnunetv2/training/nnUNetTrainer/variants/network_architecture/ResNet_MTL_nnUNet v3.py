import torch
import torch.nn as nn
from typing import Union, List, Tuple, Type

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.initialization.weight_init import (
    InitWeights_He, init_last_bn_before_add_to_0
)


class ResNet_MTL_nnUNet(ResidualEncoderUNet):
    """
    seg 分支：完全使用原版 ResidualEncoderUNet 的 encoder + UNetDecoder
    cls 分支：只读 encoder bottleneck + seg logits，做一个 MLP 分类头

    task_mode:
      - 'seg_only' : 只训 seg，cls 分支 no_grad + 参数不在 optimizer 里
      - 'both'     : seg+cls 一起训（可选）
      - 'cls_only' : 只训 cls（可选）
    """
    def __init__(
        self,
        # 这部分参数必须和 plans 里的 ResidualEncoderUNet 完全对齐
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[nn.Module],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[nn.Module]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
        bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
        stem_channels: int = None,

        # 额外：分类相关
        cls_num_classes: int = 3,
        task_mode: str = 'seg_only',
    ):
        super().__init__(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
            block=block,
            bottleneck_channels=bottleneck_channels,
            stem_channels=stem_channels,
        )

        assert task_mode in ('seg_only', 'both', 'cls_only')
        self.task_mode = task_mode

        # -------------------------
        # cls 头：简单 MLP，输入 bottleneck GAP + seg logits GAP（最高分辨率）
        # -------------------------
        self.bottleneck_channels = features_per_stage[-1]
        self.num_classes = num_classes
        self.cls_num_classes = cls_num_classes

        cls_in_dim = self.bottleneck_channels + self.num_classes  # bottleneck GAP + seg GAP
        hidden_dim = max(cls_in_dim // 2, cls_num_classes * 2)

        self.classification_head = nn.Sequential(
            nn.Linear(cls_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, cls_num_classes),
        )

        # 初始化：完全沿用原版 ResidualEncoderUNet 的风格
        InitWeights_He(1e-2)(self)
        init_last_bn_before_add_to_0(self)

    def set_task_mode(self, mode: str):
        assert mode in ('seg_only', 'both', 'cls_only')
        self.task_mode = mode

    def _forward_cls_branch(self, bottleneck_features, seg_output):
        """
        bottleneck_features: encoder 最后一层特征 [B, Cb, D, H, W]
        seg_output: decoder 输出 logits（deep supervision 时是 list，取最高分辨率那张）
        """
        if isinstance(seg_output, (list, tuple)):
            seg_logits = seg_output[0]     # [B, K, D, H, W]
        else:
            seg_logits = seg_output

        # GAP
        v_enc = bottleneck_features.mean(dim=(2, 3, 4))  # [B, Cb]
        v_seg = seg_logits.mean(dim=(2, 3, 4))          # [B, K]

        cls_vec = torch.cat([v_enc, v_seg], dim=1)      # [B, Cb+K]
        out = self.classification_head(cls_vec)         # [B, cls_num_classes]
        return out

    def forward(self, x, task_mode: str = None):
        """
        seg 分支严格保持：
            skips = self.encoder(x)
            seg = self.decoder(skips)
        这两行和父类 ResidualEncoderUNet.forward 完全一致。
        """
        mode = task_mode if task_mode is not None else self.task_mode
        assert mode in ('seg_only', 'both', 'cls_only')

        # ---- encoder + decoder：和原版一模一样 ----
        skips = self.encoder(x)
        bottleneck_features = skips[-1]
        seg_output = self.decoder(skips)   # list or tensor，完全等价于原版 forward(x)

        # ---- 根据模式决定 cls 分支如何参与 ----
        if mode == 'seg_only':
            # 只训 seg：cls 分支只 forward，无梯度（或者你可以直接不算，返回零向量）
            with torch.no_grad():
                cls_output = self._forward_cls_branch(bottleneck_features, seg_output)

        elif mode == 'cls_only':
            # 只训 cls：不让梯度回 encoder/decoder
            with torch.no_grad():
                seg_detached = seg_output
                # seg 已经 no_grad，所以不需要再 detach
                bottleneck_detached = bottleneck_features.detach()
            cls_output = self._forward_cls_branch(bottleneck_detached, seg_detached)

        else:  # 'both'
            cls_output = self._forward_cls_branch(bottleneck_features, seg_output)

        return seg_output, cls_output
