import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Type

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.initialization.weight_init import (
    InitWeights_He, init_last_bn_before_add_to_0
)
from torch.amp import autocast as autocast_cuda


class DualPathTransformerBlock(nn.Module):
    """
    轻量版 Dual-path Transformer：
      - ct_token <- mem_tokens (cross-attn)
      - mem_tokens <- ct_token (cross-attn)
      - 各自 FFN + 残差
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        ff_mult: int = 2,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # cross-attn: ct <- mem
        self.norm_ct1 = nn.LayerNorm(d_model)
        self.norm_mem1 = nn.LayerNorm(d_model)
        self.attn_ct_from_mem = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        # cross-attn: mem <- ct
        self.norm_ct2 = nn.LayerNorm(d_model)
        self.norm_mem2 = nn.LayerNorm(d_model)
        self.attn_mem_from_ct = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        # FFN for ct token
        self.ff_ct = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.ff_ct_norm = nn.LayerNorm(d_model)

        # FFN for mem tokens
        self.ff_mem = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.ff_mem_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        ct_token: torch.Tensor,   # [B, 1, D]
        mem_tokens: torch.Tensor  # [B, M, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # === 强制用 float32，避免 AMP 下 multiheadattention 溢出 ===
        # ct_token = ct_token.float()
        # mem_tokens = mem_tokens.float()

        # ----- 路径1: ct <- mem -----
        ct_res = ct_token
        mem_res = mem_tokens

        ct_norm = self.norm_ct1(ct_token)
        mem_norm = self.norm_mem1(mem_tokens)
        ct_updated, _ = self.attn_ct_from_mem(
            query=ct_norm,
            key=mem_norm,
            value=mem_norm,
            need_weights=False,
        )
        ct_token = ct_res + ct_updated

        # ----- 路径2: mem <- ct -----
        ct_norm2 = self.norm_ct2(ct_token)
        mem_norm2 = self.norm_mem2(mem_tokens)
        mem_updated, _ = self.attn_mem_from_ct(
            query=mem_norm2,
            key=ct_norm2,
            value=ct_norm2,
            need_weights=False,
        )
        mem_tokens = mem_res + mem_updated

        # ----- FFN + 残差 -----
        ct_ff = self.ff_ct(ct_token)
        ct_token = self.ff_ct_norm(ct_token + ct_ff)

        mem_ff = self.ff_mem(mem_tokens)
        mem_tokens = self.ff_mem_norm(mem_tokens + mem_ff)

        # 这里直接保持 float32 输出就行，外面 cls_out 也是 float32 计算
        return ct_token, mem_tokens


class ResNet_MTL_nnUNet(ResidualEncoderUNet):
    """
    seg 分支：完全使用原版 ResidualEncoderUNet 的 encoder + UNetDecoder
    cls 分支：
        - encoder 多尺度特征 global max pooling -> concat -> multi_scale_feat
        - multi_scale_feat -> ct_token (1 token)
        - 3 个记忆原型 token（每类 1 个）
        - Dual-path Transformer 交互
        - ct_token -> Linear 输出 3 类 logits

    task_mode:
      - 'seg_only' : 只训 seg（cls 分支 no_grad）
      - 'both'     : seg + cls 一起训
      - 'cls_only' : 只训 cls（encoder/decoder no_grad）
    """
    def __init__(
        self,
        # --- 与原版 ResidualEncoderUNet 对齐 ---
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

        # 多尺度 pooled feature 总维度
        self.cls_feat_dim = sum(self.encoder_feat_channels)

        # 将多尺度特征投到较小维度 d_model，当成一个 ct token
        self.d_model = min(256, self.cls_feat_dim)  # 控制容量，避免 240 例过拟合太严重
        self.ct_proj = nn.Linear(self.cls_feat_dim, self.d_model)

        # 3 个类的记忆原型 token（3 分类）
        # shape: [num_classes, d_model]
        self.prototype_memory = nn.Parameter(
            torch.randn(cls_num_classes, self.d_model) * 0.02
        )

        # 单层 Dual-path Transformer block
        self.dual_block = DualPathTransformerBlock(
            d_model=self.d_model,
            num_heads=4,
            ff_mult=2,
            attn_dropout=0.1,
            ffn_dropout=cls_dropout,
        )

        # 最后的分类头：直接用 ct_token 的表示做线性分类
        self.cls_out = nn.Linear(self.d_model, cls_num_classes)

        # 初始化（卷积/线性等）
        InitWeights_He(1e-2)(self)
        init_last_bn_before_add_to_0(self)
        # 原型在上面已经手动 normal 初始化了，不会被 He 覆盖

    # ---- 小工具：外部可以随时切模式 ----
    def set_task_mode(self, mode: str):
        assert mode in ('seg_only', 'both', 'cls_only')
        self.task_mode = mode

    # ---- 主 forward：seg 路径保持 nnUNet 原样 ----
    def forward(self, x, task_mode: str = None):
        mode = task_mode if task_mode is not None else self.task_mode
        assert mode in ('seg_only', 'both', 'cls_only')

        # encoder + decoder
        if mode == 'cls_only':
            # 只做分类：encoder/decoder 不要梯度
            with torch.no_grad():
                skips = self.encoder(x)      # list of [B, C_i, D_i, H_i, W_i]
                seg_output = self.decoder(skips)
        else:
            # seg_only / both：正常训练 seg 分支
            skips = self.encoder(x)
            seg_output = self.decoder(skips)

        # 分类分支

        if mode == 'seg_only':
            with torch.no_grad():
                # seg_output 已经在 no_grad 上下文中
                cls_output = self._forward_cls_branch(skips, seg_output)
        elif mode == 'cls_only':
                # 冻结 encoder 和 decoder 的特征
            skips_detached = [s.detach() for s in skips]
            # seg_output 已经在 no_grad 上下文中计算，自带 no_grad 属性
            cls_output = self._forward_cls_branch(skips_detached, seg_output)
        else:  # both
            # 梯度流经 encoder, decoder, 和 cls_branch
            cls_output = self._forward_cls_branch(skips, seg_output)

        return seg_output, cls_output

    # ---- 分类分支：多尺度 pooling + Dual-path Transformer + 记忆原型 ----
    def _forward_cls_branch(self, skips, seg_output):
        # 1. 获取 seg_logits (必须转 float32
        if isinstance(seg_output, (list, tuple)):
            seg_logits = seg_output[0].float()
        else:
            seg_logits = seg_output.float()

        # 2. 生成掩码 (并 detach!)
        probs = F.softmax(seg_logits, dim=1)
        lesion_mask_hires = probs[:, 2:3, ...].detach() # 阶段 5 修复

        # 3. Masked Pooling (feat 不再 .float())
        pooled_feats = []
        for feat in skips:
            # feat = feat.float()  <-- 已删除
            current_shape = feat.shape[2:]
            lesion_mask_resized = F.interpolate(lesion_mask_hires, 
                                                size=current_shape, 
                                                mode='trilinear', 
                                                align_corners=False)
            feat_masked = feat * lesion_mask_resized
            feat_sum_per_channel = feat_masked.sum(dim=(2, 3, 4))
            mask_sum_per_channel = lesion_mask_resized.sum(dim=(2, 3, 4)) + 1e-6
            pooled = feat_sum_per_channel / mask_sum_per_channel
            pooled_feats.append(pooled)

        # 4. 拼接 (不再 .float())
        multi_scale_feat = torch.cat(pooled_feats, dim=1)

        ct_token = self.ct_proj(multi_scale_feat)
        ct_token = ct_token.unsqueeze(1)

        B = ct_token.size(0)
        # (不再 .float())
        mem_tokens = self.prototype_memory.unsqueeze(0).expand(B, -1, -1) 

        # 5. Dual block
        ct_token, mem_tokens = self.dual_block(ct_token, mem_tokens)

        # 6. Output
        ct_token = ct_token.squeeze(1)
        cls_logits = self.cls_out(ct_token)

        # 7. 返回 float32 的 logits
        return cls_logits.float()


