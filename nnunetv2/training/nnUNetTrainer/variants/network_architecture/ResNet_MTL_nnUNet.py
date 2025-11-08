import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Type

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.initialization.weight_init import (
    InitWeights_He, init_last_bn_before_add_to_0
)


class ResNet_MTL_nnUNet(ResidualEncoderUNet):
    """
    seg 分支：完全使用原版 ResidualEncoderUNet 的 encoder + UNetDecoder
    cls 分支：PANDA 风格
        - 直接从 encoder 的多尺度特征 (skips 的各层) 做全局池化
        - 将各尺度 pooled 向量拼接
        - 通过两层 MLP 输出分类 logits

    task_mode:
      - 'seg_only' : 只训 seg（cls 分支 no_grad）
      - 'both'     : seg + cls 一起训
      - 'cls_only' : 只训 cls（encoder/decoder no_grad）
    """
    def __init__(
        self,
        # --- 这部分必须和 plans 里的 ResidualEncoderUNet 完全对齐 ---
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
        # --- 额外：分类相关 ---
        cls_num_classes: int = 3,
        task_mode: str = 'seg_only',
        cls_dropout: float = 0.3,
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
        self.cls_num_classes = cls_num_classes
        self.num_classes = num_classes  # seg 类别数（含背景）

        # --- encoder 各 stage 通道数 ---
        if isinstance(features_per_stage, int):
            feat_list = [features_per_stage] * n_stages
        else:
            feat_list = list(features_per_stage)
        assert len(feat_list) == n_stages, "features_per_stage 长度必须等于 n_stages"
        self.encoder_feat_channels = feat_list

        # PANDA 多尺度：使用所有 stage 的 encoder 特征做 global pooling
        self.cls_feat_dim = sum(self.encoder_feat_channels)

        # 两层 MLP 分类头（PANDA 风格）
        hidden_dim = max(self.cls_feat_dim // 2, cls_num_classes * 2)
        self.classification_head = nn.Sequential(
            nn.Linear(self.cls_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cls_dropout),
            nn.Linear(hidden_dim, cls_num_classes),
        )

        # 初始化：沿用 nnUNet 风格（相当于对整个网络做 He init）
        InitWeights_He(1e-2)(self)
        init_last_bn_before_add_to_0(self)

    # ---- 小工具：外部可以随时切模式 ----
    def set_task_mode(self, mode: str):
        assert mode in ('seg_only', 'both', 'cls_only')
        self.task_mode = mode

    # ---- 主 forward：严格保持 seg 路径与原生 ResidualEncoderUNet 一致 ----
    def forward(self, x, task_mode: str = None):
        """
        seg 分支严格保持：
            skips = self.encoder(x)
            seg   = self.decoder(skips)
        """
        mode = task_mode if task_mode is not None else self.task_mode
        assert mode in ('seg_only', 'both', 'cls_only')

        # ---- encoder + decoder ----
        if mode == 'cls_only':
            # 只做分类：encoder/decoder 不要梯度
            with torch.no_grad():
                skips = self.encoder(x)      # list of [B, C_i, D_i, H_i, W_i]
                seg_output = self.decoder(skips)
        else:
            # seg_only / both：正常训练 seg 分支
            skips = self.encoder(x)
            seg_output = self.decoder(skips)

        # ---- 分类分支（PANDA 多尺度全局池化） ----
        if mode == 'seg_only':
            # 只训 seg：分类头 no_grad，完全不更新 cls 参数
            with torch.no_grad():
                cls_output = self._forward_cls_branch(skips)
        elif mode == 'cls_only':
            # 只训 cls：encoder/decoder 在上面已经 no_grad，这里可以更新 cls 参数
            cls_output = self._forward_cls_branch(skips)
        else:  # both
            cls_output = self._forward_cls_branch(skips)

        return seg_output, cls_output

    # ---- PANDA 风格分类分支：多尺度 encoder 特征 + 全局 max pooling + MLP ----
    def _forward_cls_branch(self, skips: List[torch.Tensor]) -> torch.Tensor:
        """
        skips: encoder 各 stage 输出的特征列表
            len(skips) = n_stages
            每个元素形状: [B, C_i, D_i, H_i, W_i]

        做法：
          1) 对每个 stage 特征做 global max pooling -> [B, C_i]
          2) 沿通道维拼接 -> [B, sum_i C_i] = [B, cls_feat_dim]
          3) MLP 分类头 -> [B, cls_num_classes]
        """
        pooled_feats = []
        for feat in skips:
            # 自适应全局 max pooling 到 1x1x1
            # feat: [B, C, D, H, W]
            pooled = F.adaptive_max_pool3d(feat, output_size=1)  # [B, C, 1, 1, 1]
            pooled = pooled.view(pooled.size(0), -1)             # [B, C]
            pooled_feats.append(pooled)

        # 多尺度特征拼接
        multi_scale_feat = torch.cat(pooled_feats, dim=1)        # [B, cls_feat_dim]

        # MLP 分类头
        cls_logits = self.classification_head(multi_scale_feat)  # [B, cls_num_classes]
        return cls_logits
