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

class NormedLinear(nn.Linear):
    def __init__(self, in_features, out_features, s=30.):
        super().__init__(in_features, out_features, bias=False)
        self.s = s
        nn.init.xavier_normal_(self.weight)
    def forward(self, x):
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        return self.s * F.linear(x, w)
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
        ct_tokens: torch.Tensor,   # [B, N, D]，N 可以是 1 或 2
        mem_tokens: torch.Tensor   # [B, M, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # === 强制用 float32，避免 AMP 下 multiheadattention 溢出 ===
        # ct_token = ct_token.float()
        # mem_tokens = mem_tokens.float()
        orig_dtype = ct_tokens.dtype
        # ----- 路径1: ct <- mem -----
        ct_res = ct_tokens
        mem_res = mem_tokens

        # 使用更稳定且高吞吐的 BF16（若可用），以启用 SDPA/FlashAttention 快路径
        try:
            attn_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else orig_dtype
        except Exception:
            attn_dtype = orig_dtype

        ct_norm = self.norm_ct1(ct_tokens).to(attn_dtype)
        mem_norm = self.norm_mem1(mem_tokens).to(attn_dtype)
        ct_updated, _ = self.attn_ct_from_mem(query=ct_norm, key=mem_norm, value=mem_norm, need_weights=False)
        ct_tokens = ct_res + ct_updated.to(orig_dtype)

        # ----- 路径2: mem <- ct -----
        ct_norm2 = self.norm_ct2(ct_tokens).to(attn_dtype)
        mem_norm2 = self.norm_mem2(mem_tokens).to(attn_dtype)
        mem_updated, _ = self.attn_mem_from_ct(query=mem_norm2, key=ct_norm2, value=ct_norm2, need_weights=False)
        mem_tokens = mem_res + mem_updated.to(orig_dtype)

        # ----- FFN + 残差 -----
        ct_ff = self.ff_ct(ct_tokens)
        ct_tokens = self.ff_ct_norm(ct_tokens + ct_ff)

        mem_ff = self.ff_mem(mem_tokens)
        mem_tokens = self.ff_mem_norm(mem_tokens + mem_ff)

        # 这里直接保持 float32 输出就行，外面 cls_out 也是 float32 计算
        return ct_tokens, mem_tokens


class ResNet_MTL_nnUNet(ResidualEncoderUNet):
    """
    seg 分支：完全使用原版 ResidualEncoderUNet 的 encoder + UNetDecoder
    cls 分支：
        - 直接使用“原图强度”在两类 ROI 上的统计：
          • 病灶 ROI（lesion），轻微膨胀
          • 胰腺 ROI（film，经轻膨胀/可选闭运算）
        - 分别对两路 ROI 做软加权平均（不再 Top‑K），得到两个向量
        - 线性投影 -> 两个 ct tokens (t_les, t_pan)
        - 3 个记忆原型 token（每类 1 个）
        - Dual-path Transformer 双向交互（mem↔ct），query 维度 [B,2,D]
        - 对两个 ct token 自适应加权聚合 -> 分类 logits

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
        pancreas_channel_idx: int = 1,
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
        # 记录病灶/胰腺通道索引，供 ROI 选择使用
        self.lesion_channel_idx = lesion_channel_idx
        self.pancreas_channel_idx = pancreas_channel_idx

        # --- encoder 各 stage 通道数 ---
        if isinstance(features_per_stage, int):
            feat_list = [features_per_stage] * n_stages
        else:
            feat_list = list(features_per_stage)
        assert len(feat_list) == n_stages, "features_per_stage 长度必须等于 n_stages"
        self.encoder_feat_channels = feat_list

        # 多尺度 pooled feature 总维度（原 nnUNet encoder 通道总和）
        self.cls_feat_dim = sum(self.encoder_feat_channels)

        # Perceiver-style 分类分支超参
        self.token_grid = (2, 2, 2)  # 每个 ROI 的 3D 网格划分
        self.min_cell_occ = 0.01     # 网格占比门控阈值（避免近空格子噪声 token）
        self.d_model = 128           # 统一投影维度 D，建议 128/192
        self.num_latent = 16         # latent token 个数 L
        self.num_heads = 4           # 注意力头数，需整除 D

        # 原图 ROI -> D 的线性投影（两路：lesion / pancreas）
        self.ct_proj_les = nn.Linear(input_channels, self.d_model)
        self.ct_proj_ctx = nn.Linear(input_channels, self.d_model)

        # 每个 encoder stage 单独的线性投影 C_s -> D（应用于网格 token）
        self.enc_proj = nn.ModuleList([
            nn.Linear(c, self.d_model) for c in self.encoder_feat_channels
        ])

        # 源/层/网格位置编码
        # 0: enc_L, 1: enc_Ctx, 2: dec_L(预留), 3: dec_Ctx(预留), 4: raw_L, 5: raw_P
        self.src_embed = nn.Embedding(6, self.d_model)
        self.stage_embed_enc = nn.Embedding(len(self.encoder_feat_channels), self.d_model)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, self.d_model), nn.GELU(), nn.Linear(self.d_model, self.d_model)
        )
        self.cls_topk = cls_topk  # 例如 0.1 表示 ROI 体素的前 10%
        # 3 个类的记忆原型 token（可作为额外 memory tokens 附加进 Token Bank）
        self.prototype_memory = nn.Parameter(torch.randn(cls_num_classes, self.d_model) * 0.02)
        # 关闭 Top-K ROI 池化：改为原图 ROI 的加权平均，不再做 Top-K 选择
        self.use_topk_pool = False
        # 仍保留接口字段，但不再生效
        self.topk_ratio_lesion = 0.0
        self.topk_ratio_ctx = 0.0
        # 分类 ROI 轻微膨胀，默认开启（可被外部覆盖）
        self.cls_roi_dilate = True
        # 胰腺 “film” ROI 的轻膨胀/闭运算参数
        self.film_dilate_kernel = 3
        self.film_dilate_iters = 1
        # 使用闭运算（dilate->erode）的可微近似来生成平滑外膜，默认开启
        self.film_use_closing = True

        # 多层 Dual-path Transformer blocks（Perceiver-style：latent↔memory）
        self.blocks = nn.ModuleList([
            DualPathTransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ff_mult=2,
                attn_dropout=0.1,
                ffn_dropout=cls_dropout,
            )
            for _ in range(2)
        ])
        # latent tokens（[L, D]）
        self.latent = nn.Parameter(torch.randn(self.num_latent, self.d_model) * 0.02)

        # 最后的分类头：直接用 ct_token 的表示做线性分类
        # self.cls_out = nn.Linear(self.d_model, cls_num_classes)
        # 归一化余弦分类器”（CosFace/ArcFace 风格）
        self.cls_out = NormedLinear(self.d_model, cls_num_classes, s=30.)
        # 旧的 α 融合已不再使用；保留分类头为 NormedLinear

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
                cls_output = self._forward_cls_branch(x, skips, seg_output, roi_mask)
        elif mode == 'cls_only':
                # 冻结 encoder 和 decoder 的特征
            skips_detached = [s.detach() for s in skips]
            cls_output = self._forward_cls_branch(x, skips_detached, seg_output, roi_mask)
        else:  # both
            # 梯度流经 encoder, decoder, 和 cls_branch
            cls_output = self._forward_cls_branch(x, skips, seg_output, roi_mask)

        # 返回分割 logits 和分类 logits（分类为每病例 [B, C_cls]）
        return seg_output, cls_output

    # ---- 分类分支：多尺度 pooling + Dual-path Transformer + 记忆原型 ----
    def _forward_cls_branch(self, x, skips, seg_output, roi_mask: torch.Tensor = None):
        if roi_mask is not None:
            roi_mask = roi_mask.float().detach()
            if roi_mask.ndim == 4:
                roi_mask = roi_mask.unsqueeze(1)  # B,1,D,H,W
            elif roi_mask.ndim != 5:
                raise RuntimeError(f"roi_mask must be 4D or 5D, got {roi_mask.ndim}")

            # lesion from roi_mask[: ,0]
            lesion_mask_hires = roi_mask[:, 0:1]
            if getattr(self, 'cls_roi_dilate', False):
                lesion_mask_hires = F.max_pool3d(lesion_mask_hires, kernel_size=3, stride=1, padding=1)

            if roi_mask.shape[1] >= 2:
                # 存在 GT 胰腺：对该通道做 3x3x3 轻膨胀即可
                pancreas_mask_hires = roi_mask[:, 1:2]
                pancreas_mask_hires = F.max_pool3d(pancreas_mask_hires, kernel_size=3, stride=1, padding=1)
            else:
                # 胰腺 ROI 来自网络分割概率并做闭运算（平滑外膜）
                seg_logits_tmp = (seg_output[0] if isinstance(seg_output, (list, tuple)) else seg_output).float()
                probs_tmp = F.softmax(seg_logits_tmp, dim=1)
                p_idx = int(max(0, min(getattr(self, 'pancreas_channel_idx', 1), probs_tmp.shape[1] - 1)))
                pancreas_mask_hires = probs_tmp[:, p_idx:p_idx + 1, ...].detach()
                pancreas_mask_hires = self._make_film_roi(pancreas_mask_hires)
        else:
            seg_logits = (seg_output[0] if isinstance(seg_output, (list, tuple)) else seg_output).float()
            probs = F.softmax(seg_logits, dim=1)
            # 使用“病灶”通道作为 ROI，而非 1 - P(bg)
            lesion_idx = getattr(self, 'lesion_channel_idx', probs.shape[1] - 1)
            lesion_idx = int(max(0, min(lesion_idx, probs.shape[1] - 1)))
            lesion_mask_hires = probs[:, lesion_idx:lesion_idx + 1, ...].detach()
            if getattr(self, 'cls_roi_dilate', False):
                lesion_mask_hires = F.max_pool3d(lesion_mask_hires, kernel_size=3, stride=1, padding=1)
            # 胰腺 film ROI 来自 P 概率并轻膨胀/闭运算
            pancreas_idx = int(max(0, min(getattr(self, 'pancreas_channel_idx', 1), probs.shape[1] - 1)))
            pancreas_mask_hires = probs[:, pancreas_idx:pancreas_idx + 1, ...].detach()
            pancreas_mask_hires = self._make_film_roi(pancreas_mask_hires)

        # Perceiver-style：构建 Token Bank（enc L/ctx + 原图 L/P），latent 交互聚合
        x_img = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)  # (B,C,D,H,W)
        B = x_img.shape[0]

        tokens_list = []  # [B, n_i, D]

        # 1) Encoder stages: ROI 网格 token（L/Ctx），保符号 Masked-GeM
        gD, gH, gW = self.token_grid
        grid_z = torch.linspace(-1, 1, steps=gD, device=x_img.device)
        grid_y = torch.linspace(-1, 1, steps=gH, device=x_img.device)
        grid_x = torch.linspace(-1, 1, steps=gW, device=x_img.device)
        gz, gy, gx = torch.meshgrid(grid_z, grid_y, grid_x, indexing='ij')
        pos = torch.stack([gz, gy, gx], dim=-1).view(1, gD * gH * gW, 3)  # [1, n_cells, 3]
        pos_emb = self.pos_mlp(pos)  # [1, n_cells, D]

        for s, feat in enumerate(skips):
            feat = torch.nan_to_num(feat.float(), nan=0.0, posinf=0.0, neginf=0.0)
            roi_L = F.interpolate(lesion_mask_hires, size=feat.shape[2:], mode='trilinear', align_corners=False)
            roi_C = F.interpolate((pancreas_mask_hires * (1.0 - lesion_mask_hires)).clamp_min(0.0),
                                   size=feat.shape[2:], mode='trilinear', align_corners=False)

            tL, occL = self.roi_grid_tokens(feat, roi_L, grid=self.token_grid, p=3.0)   # t: [B,n,Cs], occ: [B,n,1]
            tC, occC = self.roi_grid_tokens(feat, roi_C, grid=self.token_grid, p=2.5)

            tL = self.enc_proj[s](tL) + self.src_embed.weight[0].view(1, 1, -1) + self.stage_embed_enc.weight[s].view(1, 1, -1) + pos_emb
            tC = self.enc_proj[s](tC) + self.src_embed.weight[1].view(1, 1, -1) + self.stage_embed_enc.weight[s].view(1, 1, -1) + pos_emb
            # 软门控：近空格子（ROI 占比过低）置零，避免噪声 token 分散注意力
            thr = float(getattr(self, 'min_cell_occ', 0.01))
            gateL = (occL >= thr).to(dtype=tL.dtype)
            gateC = (occC >= thr).to(dtype=tC.dtype)
            tL = tL * gateL
            tC = tC * gateC
            tokens_list += [tL, tC]

        # 2) 原图 ROI 全局 tokens（保符号 Masked-GeM）
        raw_L = self.masked_gem_signed(x_img, lesion_mask_hires, p=3.0)    # [B,C_in]
        raw_P = self.masked_gem_signed(x_img, pancreas_mask_hires, p=2.0)  # [B,C_in]
        raw_L = self.ct_proj_les(raw_L).unsqueeze(1) + self.src_embed.weight[4].view(1, 1, -1)
        raw_P = self.ct_proj_ctx(raw_P).unsqueeze(1) + self.src_embed.weight[5].view(1, 1, -1)
        tokens_list += [raw_L, raw_P]

        # 3) 拼接 Token Bank（memory tokens）
        mem_tokens = torch.cat(tokens_list, dim=1)  # [B, N, D]

        # 可选：附加 prototype 作为额外 memory tokens
        proto = F.normalize(self.prototype_memory.float(), dim=-1).unsqueeze(0).expand(B, -1, -1)
        mem_tokens = torch.cat([mem_tokens, proto], dim=1)

        # 4) latent tokens（ct tokens），多层 cross-attn 聚合
        ct_tokens = self.latent.unsqueeze(0).expand(B, -1, -1).float()  # [B, L, D]
        for blk in self.blocks:
            ct_tokens, mem_tokens = blk(ct_tokens, mem_tokens)

        # 5) 分类：平均聚合 latent
        ct_agg = ct_tokens.mean(dim=1)
        cls_logits = self.cls_out(ct_agg)

        return cls_logits.float()

    def _make_film_roi(self, p_mask: torch.Tensor) -> torch.Tensor:
        """根据胰腺概率图生成 soft 'film' ROI。
        - 轻膨胀：max_pool3d kernel=film_dilate_kernel, iters=film_dilate_iters
        - 可选闭运算：dilate->erode（对 soft map 采用 max_pool 与其对偶实现）
        """
        k = int(getattr(self, 'film_dilate_kernel', 3))
        iters = int(getattr(self, 'film_dilate_iters', 1))
        use_closing = bool(getattr(self, 'film_use_closing', False))
        if k < 1:
            return p_mask
        pad = k // 2
        out = p_mask
        for _ in range(max(1, iters)):
            out = F.max_pool3d(out, kernel_size=k, stride=1, padding=pad)
        if use_closing:
            # erosion 近似：对 -x 做 max_pool 即 min_pool
            tmp = out
            for _ in range(max(1, iters)):
                tmp = -F.max_pool3d(-tmp, kernel_size=k, stride=1, padding=pad)
            out = tmp
        return out

    @staticmethod
    def _binarize_soft_mask(m: torch.Tensor, thr: float = 0.2) -> torch.Tensor:
        return (m >= float(thr)).to(dtype=m.dtype)


    @staticmethod
    def masked_gem_signed(x: torch.Tensor, m: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
        """
        保符号 Masked-GeM：对未 ReLU 的特征更稳。
        x: [B,C,D,H,W], m: [B,1,D,H,W]
        返回: [B,C]
        """
        B, C = x.shape[:2]
        x = x.view(B, C, -1).float()
        m = m.view(B, 1, -1).float()
        p_ = torch.as_tensor(p, dtype=x.dtype, device=x.device).view(1, 1, 1)
        w = (m * (x.abs() + eps).pow(p_ - 1)).clamp_min(eps)  # (B,1,N) 广播到 (B,C,N)
        num = (w * x).sum(-1)                                  # (B,C)
        den = w.sum(-1).clamp_min(eps)                         # (B,1)
        return num / den

    def roi_grid_tokens(self, feat: torch.Tensor, roi: torch.Tensor, grid=(2, 2, 2), p: float = 3.0, eps: float = 1e-6):
        """
        基于 ROI 的保符号 Masked-GeM + 3D 网格划分，生成多个 token，并返回每格 ROI 占比用于门控。
        feat: [B,C,D,H,W], roi: [B,1,D,H,W]
        返回: out: [B, n_cells, C], occ: [B, n_cells, 1]（∈[0,1]）
        """
        B, C = feat.shape[:2]
        gD, gH, gW = grid
        x = feat.float()
        m = roi.float()
        w = (x.abs() + eps).pow(p - 1) * m                      # [B,C,D,H,W]
        num = F.adaptive_avg_pool3d(w * x, output_size=(gD, gH, gW))  # [B,C,gD,gH,gW]
        den = F.adaptive_avg_pool3d(w,     output_size=(gD, gH, gW)).clamp_min(eps)
        out = (num / den).view(B, C, -1).transpose(1, 2)        # [B, n_cells, C]
        # 每格 ROI 占比（范围 0-1）
        occ = F.adaptive_avg_pool3d(m, output_size=(gD, gH, gW)).view(B, -1, 1)
        return out, occ

    @staticmethod
    def topk_pool(feats: torch.Tensor, mask: torch.Tensor, k: float = 0.1, bin_mask: torch.Tensor = None) -> torch.Tensor:
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
        mb = bin_mask.view(B, 1, -1) if bin_mask is not None else m

        # 计算每个样本的 k
        if isinstance(k, float) and k <= 1.0:
            k_each = torch.clamp((mb.sum(-1) * k).long(), min=1)  # (B,1)
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
