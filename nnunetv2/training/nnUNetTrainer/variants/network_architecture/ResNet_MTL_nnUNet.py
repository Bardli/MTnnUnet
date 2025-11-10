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
        orig_dtype = ct_token.dtype
        # ----- 路径1: ct <- mem -----
        ct_res = ct_token
        mem_res = mem_tokens

        ct_norm = self.norm_ct1(ct_token).float()
        mem_norm = self.norm_mem1(mem_tokens).float()
        ct_updated, _ = self.attn_ct_from_mem(query=ct_norm, key=mem_norm, value=mem_norm, need_weights=False)
        ct_token = ct_res + ct_updated.to(orig_dtype)

        # ----- 路径2: mem <- ct -----
        ct_norm2 = self.norm_ct2(ct_token).float()
        mem_norm2 = self.norm_mem2(mem_tokens).float()
        mem_updated, _ = self.attn_mem_from_ct(query=mem_norm2, key=ct_norm2, value=ct_norm2, need_weights=False)
        mem_tokens = mem_res + mem_updated.to(orig_dtype)

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
      - 'seg_only' : 只训 seg（cls 分支 no_grad，不更新）
      - 'both'     : seg + voxel-level cls 一起训（推荐）
      - 'cls_only' : 只训 cls（encoder/decoder no_grad，仅更新 cls head）
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
        # === 额外：分类相关 ===
        cls_num_classes: int = 3,
        task_mode: str = 'seg_only',
        cls_dropout: float = 0.3,
        lesion_channel_idx: int = 2,
        cls_topk: float = 0.1
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
        self.cls_topk = cls_topk  # 例如 0.1 表示 ROI 体素的前 10%
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

    # 方便外部切换模式
    def set_task_mode(self, mode: str):
        assert mode in ('seg_only', 'both', 'cls_only')
        self.task_mode = mode

    # ---- 主 forward：seg 路径保持 nnUNet 原样 ----
    def forward(self, x, task_mode: str = None, roi_mask: torch.Tensor = None):
        mode = task_mode if task_mode is not None else self.task_mode
        assert mode in ('seg_only', 'both', 'cls_only')

        # encoder + decoder
        if mode == 'cls_only':
            # 分类微调阶段：encoder/decoder 冻结
            with torch.no_grad():
                skips = self.encoder(x)      # list of [B, C_i, D_i, H_i, W_i]
                seg_output = self.decoder(skips)
        else:
            skips = self.encoder(x)
            seg_output = self.decoder(skips)

        # 分类分支

        if mode == 'seg_only':
            with torch.no_grad():
                # seg_output 已经在 no_grad 上下文中
                cls_output = self._forward_cls_branch(skips, seg_output, roi_mask)
        elif mode == 'cls_only':
                # 冻结 encoder 和 decoder 的特征
            skips_detached = [s.detach() for s in skips]
            cls_output = self._forward_cls_branch(skips_detached, seg_output, roi_mask)
        else:  # both
            # 梯度流经 encoder, decoder, 和 cls_branch
            cls_output = self._forward_cls_branch(skips, seg_output, roi_mask)

        return seg_output, cls_voxel_logits  # seg_output: list/tensor; cls_voxel_logits: [B, C_cls, D,H,W]

    # ---- 分类分支：多尺度 pooling + Dual-path Transformer + 记忆原型 ----
    def _forward_cls_branch(self, skips, seg_output, roi_mask: torch.Tensor = None):
        if roi_mask is not None:
            lesion_mask_hires = roi_mask.float().detach()
            if lesion_mask_hires.ndim == 4:
                lesion_mask_hires = lesion_mask_hires.unsqueeze(1)
            elif lesion_mask_hires.ndim != 5:
                raise RuntimeError(f"roi_mask must be 4D or 5D, got {lesion_mask_hires.ndim}")
        else:
            seg_logits = (seg_output[0] if isinstance(seg_output, (list, tuple)) else seg_output).float()
            probs = F.softmax(seg_logits, dim=1)
            lesion_mask_hires = (1.0 - probs[:, 0:1, ...]).detach()
            if getattr(self, 'cls_roi_dilate', False):
                lesion_mask_hires = F.max_pool3d(lesion_mask_hires, kernel_size=3, stride=1, padding=1)

        device_type = skips[0].device.type
        with torch.autocast(device_type=device_type, enabled=False):
            pooled_feats = []
            B = skips[0].shape[0]

            use_topk = getattr(self, 'use_topk_pool', True)
            topk_ratio = getattr(self, 'topk_ratio', 0.1)

            for feat in skips:
                feat = torch.nan_to_num(feat.float(), nan=0.0, posinf=0.0, neginf=0.0)
                roi_resized = F.interpolate(
                    torch.nan_to_num(lesion_mask_hires.float(), nan=0.0, posinf=0.0, neginf=0.0),
                    size=feat.shape[2:], mode='trilinear', align_corners=False
                )

                # ROI 面积；阈值=16像素（过小则认为不可靠）
                area = roi_resized.sum(dim=(2, 3, 4))
                valid = (area > 16).squeeze(1)

                if use_topk and valid.any():
                    # 只在有效样本上做 top‑k，其他回退 GAP
                    tk_all = self.topk_pool(feat, roi_resized, k=topk_ratio)  # [B,C]
                    gap_all = feat.mean(dim=(2, 3, 4))                        # [B,C]
                    pooled = torch.where(valid.view(B, 1), tk_all, gap_all)
                else:
                    pooled = feat.mean(dim=(2, 3, 4))

                pooled_feats.append(pooled)

            multi_scale_feat = torch.cat(pooled_feats, dim=1)  # (B, sum(C_i))
            ct_token = self.ct_proj(multi_scale_feat).unsqueeze(1)       # (B,1,D)
            mem_tokens = self.prototype_memory.unsqueeze(0).expand(ct_token.size(0), -1, -1).float()

            ct_token, mem_tokens = self.dual_block(ct_token, mem_tokens)
            cls_logits = self.cls_out(ct_token.squeeze(1))

        return cls_logits.float()


    @staticmethod
    def topk_pool(feats: torch.Tensor, mask: torch.Tensor, k: float = 0.1) -> torch.Tensor:
        """
        feats: [B,C,D,H,W]；mask: [B,1,D,H,W]，元素∈[0,1]
        k∈(0,1] 表示取 ROI 体素数的 k 比例；k>=1 表示固定取前 k 个。
        注意：若 mask 全 0，返回 0 向量（调用处要自行 fallback 到 GAP）
        """
        assert feats.shape[0] == mask.shape[0] and feats.ndim == mask.ndim
        x = feats * mask                  # 屏蔽 ROI 外
        B, C = x.shape[:2]
        x = x.view(B, C, -1)              # (B,C,N)
        m = mask.view(B, 1, -1)           # (B,1,N)

        # 计算每个样本的 k
        if isinstance(k, float) and k <= 1.0:
            k_each = torch.clamp((m.sum(-1) * k).long(), min=1)  # (B,1)
        else:
            k_each = torch.full((B, 1), int(k), dtype=torch.long, device=x.device)

        kmax = int(k_each.max().item())
        if kmax <= 0 or x.shape[-1] == 0:
            return torch.zeros(B, C, device=x.device, dtype=x.dtype)

        vals, _ = torch.topk(x, k=kmax, dim=-1)  # (B,C,kmax)
        out = []
        for b in range(B):
            kb = int(k_each[b].item())
            out.append(vals[b, :, :kb].mean(-1))
        return torch.stack(out, 0)  # (B,C)

