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
    cls 分支：读取 encoder 倒数两层特征 + seg 最终 logits，做 mask 引导 + attention 融合，再接两层 MLP 分类头

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
        self.num_classes = num_classes  # seg 类别数（含背景），后面做 mask 用

        # --- encoder multi-scale 通道数 ---
        if isinstance(features_per_stage, int):
            feat_list = [features_per_stage] * n_stages
        else:
            feat_list = list(features_per_stage)
        assert len(feat_list) >= 2, "需要至少两层 encoder 以做 multi-scale 分类特征"
        enc_ch_last = feat_list[-1]
        enc_ch_second_last = feat_list[-2]

        # 分类特征维度 = 倒数两层 encoder 通道之和
        self.cls_feat_dim = enc_ch_last + enc_ch_second_last

        # attention：输入 [f_global, f_region] 拼接后的 2*dim，输出两个权重（global / region）
        self.attention_fuse = nn.Sequential(
            nn.Linear(self.cls_feat_dim * 2, 2),
            nn.Softmax(dim=1)
        )

        # 两层 MLP 分类头
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
                skips = self.encoder(x)
                seg_output = self.decoder(skips)
        else:
            # seg_only / both：正常训练 seg 分支
            skips = self.encoder(x)
            seg_output = self.decoder(skips)

        bottleneck_features = skips[-1]   # 最深层
        second_last_features = skips[-2]  # 倒数第二层

        # ---- 分类分支 ----
        if mode == 'seg_only':
            # 只训 seg：分类头 no_grad，完全不更新 cls 参数
            with torch.no_grad():
                cls_output = self._forward_cls_branch(
                    bottleneck_features,
                    second_last_features,
                    seg_output
                )
        elif mode == 'cls_only':
            # 只训 cls：encoder/decoder 在上面已经 no_grad，这里可以直接 forward
            cls_output = self._forward_cls_branch(
                bottleneck_features,
                second_last_features,
                seg_output
            )
        else:  # both
            cls_output = self._forward_cls_branch(
                bottleneck_features,
                second_last_features,
                seg_output
            )

        return seg_output, cls_output

    # ---- 分类分支：multi-scale + mask 引导 + attention 融合 ----
    def _forward_cls_branch(self,
                            bottleneck_features: torch.Tensor,
                            second_last_features: torch.Tensor,
                            seg_output):
        """
        bottleneck_features: encoder 最后一层特征 [B, Cb, D_b, H_b, W_b]
        second_last_features: encoder 倒数第二层特征 [B, Cs, D_s, H_s, W_s]
        seg_output: decoder 输出 logits（deep supervision 时为 list/tuple）
        """
        # 1) 取最终 seg logits（高分辨率）
        if isinstance(seg_output, (list, tuple)):
            seg_logits = seg_output[0]   # nnUNet deep supervision: index 0 通常是最高分辨率
        else:
            seg_logits = seg_output      # [B, K, D, H, W]

        # 2) multi-scale global pooling（不看 mask）
        f_global_last = bottleneck_features.mean(dim=(2, 3, 4))      # [B, Cb]
        f_global_second = second_last_features.mean(dim=(2, 3, 4))   # [B, Cs]
        f_global = torch.cat([f_global_last, f_global_second], dim=1)  # [B, cls_feat_dim]

        # 3) 从 seg logits 生成前景概率掩膜
        if self.num_classes > 1:
            prob = torch.softmax(seg_logits, dim=1)                  # [B, K, D, H, W]
            # 假设通道 1..K-1 都是 foreground，全部加起来
            if prob.shape[1] > 1:
                fg_prob = prob[:, 1:, ...].sum(dim=1, keepdim=True)  # [B,1,D,H,W]
            else:
                fg_prob = prob                                       # 退化情况
        else:
            # 单通道 seg（logits），用 sigmoid
            fg_prob = torch.sigmoid(seg_logits)                      # [B,1,D,H,W]（K=1）

        # 4) 下采样 mask 到 encoder 倒数 1/2 层
        B, _, D_b, H_b, W_b = bottleneck_features.shape
        _, _, D_s, H_s, W_s = second_last_features.shape

        mask_last = F.interpolate(
            fg_prob, size=(D_b, H_b, W_b),
            mode='trilinear', align_corners=False
        )
        mask_second = F.interpolate(
            fg_prob, size=(D_s, H_s, W_s),
            mode='trilinear', align_corners=False
        )

        # 5) mask 引导的平均池化（只在前景区域求均值）
        def masked_avg(feat: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
            masked = feat * mask
            num = masked.sum(dim=(2, 3, 4))          # [B,C]
            den = mask.sum(dim=(2, 3, 4)).clamp_min(eps)  # [B,1]
            return num / den

        f_region_last = masked_avg(bottleneck_features, mask_last)       # [B, Cb]
        f_region_second = masked_avg(second_last_features, mask_second)  # [B, Cs]
        f_region = torch.cat([f_region_last, f_region_second], dim=1)    # [B, cls_feat_dim]

        # 6) attention 融合 global / region 两路特征
        #    根据 concat([f_global, f_region]) 预测 [w_global, w_region]，softmax 后加权求和
        concat_features = torch.cat([f_global, f_region], dim=1)         # [B, 2*cls_feat_dim]
        weights = self.attention_fuse(concat_features)                   # [B,2]
        w_global = weights[:, 0:1]                                       # [B,1]
        w_region = weights[:, 1:2]                                       # [B,1]
        fused = w_global * f_global + w_region * f_region                # [B, cls_feat_dim]

        # 7) MLP 分类头
        cls_logits = self.classification_head(fused)                     # [B, cls_num_classes]
        return cls_logits
