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
    seg 分支：完全保持 ResidualEncoderUNet 原版 encoder + decoder（输出 seg logits）
    cls 分支：多尺度 + 深一点的 voxel-level decoder，
              读取 encoder 最后两层 + seg 最终 logits，输出 [B, cls_num_classes, D, H, W]

    task_mode:
      - 'seg_only' : 只训 seg（cls 分支 no_grad，不更新）
      - 'both'     : seg + voxel-level cls 一起训（推荐）
      - 'cls_only' : 只训 cls（encoder/decoder no_grad，仅更新 cls head）
    """
    def __init__(
        self,
        # === 与 plans 中 ResidualEncoderUNet 对齐的参数 ===
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
        # === 额外：分类相关 ===
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

        self.cls_num_classes = cls_num_classes   # subtype 个数
        self.num_classes = num_classes           # seg 类别数（含背景）

        # -----------------------------
        # 1) 多尺度通道数：encoder 最后两层
        # -----------------------------
        if isinstance(features_per_stage, int):
            feat_list = [features_per_stage] * n_stages
        else:
            feat_list = list(features_per_stage)
        self.enc_ch_last = feat_list[-1]         # bottleneck
        self.enc_ch_second_last = feat_list[-2]  # 倒数第二层

        # -----------------------------
        # 2) 各尺度的 1x1 adapter
        #    - bottleneck
        #    - encoder[-2]
        #    - seg logits
        # -----------------------------
        # 所有 cls decoder 里用的中间通道数（你想让它 overfit，可以适当大一点）
        cls_base_ch = 64  # 比 64 稍微大一点
        self.cls_base_ch = cls_base_ch

        # bottleneck -> cls_base_ch
        self.cls_enc_last_adapter = conv_op(
            self.enc_ch_last, cls_base_ch,
            kernel_size=1, stride=1, padding=0, bias=conv_bias
        )
        # encoder 倒数第二层 -> cls_base_ch
        self.cls_enc_second_adapter = conv_op(
            self.enc_ch_second_last, cls_base_ch,
            kernel_size=1, stride=1, padding=0, bias=conv_bias
        )
        # seg logits -> cls_base_ch_seg，使 seg 也能作为一层 “高分辨率 skip feature”
        self.cls_seg_adapter = conv_op(
            num_classes, cls_base_ch,
            kernel_size=1, stride=1, padding=0, bias=conv_bias
        )

        # -----------------------------
        # 3) 构造一个“稍微深一点”的 cls decoder
        #    三个 stage，每个 stage 两个 3x3 conv (带 norm+nonlin)
        # -----------------------------
        def make_block(in_ch, out_ch):
            layers: List[nn.Module] = [
                conv_op(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=conv_bias)
            ]
            if norm_op is not None:
                layers.append(norm_op(out_ch, **(norm_op_kwargs or {})))
            if nonlin is not None:
                layers.append(nonlin(**(nonlin_kwargs or {})))

            layers.append(
                conv_op(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=conv_bias)
            )
            if norm_op is not None:
                layers.append(norm_op(out_ch, **(norm_op_kwargs or {})))
            if nonlin is not None:
                layers.append(nonlin(**(nonlin_kwargs or {})))

            return nn.Sequential(*layers)

        # stage1: 只用 bottleneck (已经 upsample + 1x1)
        self.cls_block1 = make_block(cls_base_ch, cls_base_ch)

        # stage2: concat stage1 输出 + encoder[-2] (up + 1x1)
        self.cls_block2 = make_block(cls_base_ch + cls_base_ch, cls_base_ch)

        # stage3: concat stage2 输出 + seg feature (1x1)
        self.cls_block3 = make_block(cls_base_ch + cls_base_ch, cls_base_ch)

        # dropout + 最终 1x1 输出 subtype logits
        self.cls_dropout = nn.Dropout3d(p=cls_dropout)
        self.cls_out = conv_op(cls_base_ch, cls_num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        # --- 初始化：nnU-Net 风格 ---
        InitWeights_He(1e-2)(self)
        init_last_bn_before_add_to_0(self)

    # 方便外部切换模式
    def set_task_mode(self, mode: str):
        assert mode in ('seg_only', 'both', 'cls_only')
        self.task_mode = mode

    # ========================
    # 主 forward: seg + voxel cls
    # ========================
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
            # 分类微调阶段：encoder/decoder 冻结
            with torch.no_grad():
                skips = self.encoder(x)
                seg_output = self.decoder(skips)
        else:
            skips = self.encoder(x)
            seg_output = self.decoder(skips)

        bottleneck_features = skips[-1]         # [B, C_b, D_b, H_b, W_b]
        second_last_features = skips[-2]        # [B, C_s, D_s, H_s, W_s]

        # ---- voxel-level cls 分支 ----
        if mode == 'seg_only':
            with torch.no_grad():
                cls_voxel_logits = self._forward_cls_branch(
                    bottleneck_features=bottleneck_features,
                    second_last_features=second_last_features,
                    seg_output=seg_output
                )
        else:  # 'both' or 'cls_only'
            cls_voxel_logits = self._forward_cls_branch(
                bottleneck_features=bottleneck_features,
                second_last_features=second_last_features,
                seg_output=seg_output
            )

        return seg_output, cls_voxel_logits  # seg_output: list/tensor; cls_voxel_logits: [B, C_cls, D,H,W]

    # ========================
    # 辅助函数：取最终 seg logits（最高分辨率）
    # ========================
    def _get_final_seg_logits(self, seg_output):
        if isinstance(seg_output, (list, tuple)):
            seg_logits = seg_output[0]
        else:
            seg_logits = seg_output
        return seg_logits  # [B, num_classes, D, H, W]

    # ========================
    # 深一点的 voxel-level 分类支路
    # ========================
    def _forward_cls_branch(
        self,
        bottleneck_features: torch.Tensor,   # [B, C_b, D_b, H_b, W_b]
        second_last_features: torch.Tensor,  # [B, C_s, D_s, H_s, W_s]
        seg_output,
    ) -> torch.Tensor:
        # 1) seg logits（最高分辨率）
        seg_logits = self._get_final_seg_logits(seg_output)  # [B, num_classes, D, H, W]
        B, _, D, H, W = seg_logits.shape

        # 2) 把 encoder bottleneck / 倒二层上采样到 seg 尺度 + 1x1 adapter
        bottleneck_up = F.interpolate(
            bottleneck_features, size=(D, H, W),
            mode='trilinear', align_corners=False
        )
        bottleneck_up = self.cls_enc_last_adapter(bottleneck_up)      # [B, C_base, D,H,W]

        second_up = F.interpolate(
            second_last_features, size=(D, H, W),
            mode='trilinear', align_corners=False
        )
        second_up = self.cls_enc_second_adapter(second_up)            # [B, C_base, D,H,W]

        # 3) seg logits 也做 1x1 变成特征
        seg_feat = self.cls_seg_adapter(seg_logits)                   # [B, C_base, D,H,W]

        # 4) 深一点的 cls decoder
        # stage1: 只看 bottleneck 特征
        x = self.cls_block1(bottleneck_up)                            # [B, C_base, D,H,W]

        # stage2: concat encoder 倒二层
        x = torch.cat([x, second_up], dim=1)                          # [B, 2*C_base, D,H,W]
        x = self.cls_block2(x)                                        # [B, C_base, D,H,W]

        # stage3: concat seg feature（高分辨率边界信息）
        x = torch.cat([x, seg_feat], dim=1)                           # [B, 2*C_base, D,H,W]
        x = self.cls_block3(x)                                        # [B, C_base, D,H,W]

        # 5) dropout + 1x1 输出 subtype logits
        x = self.cls_dropout(x)
        cls_voxel_logits = self.cls_out(x)                            # [B, cls_num_classes, D,H,W]

        return cls_voxel_logits
