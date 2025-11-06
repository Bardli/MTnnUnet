import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, List, Type

# ---------------------------------------------------------------------
# 1. 导入 nnU-Net v2 的核心构建模块
# ---------------------------------------------------------------------
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks, BasicBlockD
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp

# ---------------------------------------------------------------------
# 2. （保留但当前没用的 CrossAttentionPooling / ClassificationHead）
#    你现在的实现没有用到 transformer，可以先留在文件里不影响
# ---------------------------------------------------------------------
class CrossAttentionPooling(nn.Module):
    def __init__(self, embed_dim, query_num, num_classes, num_heads=4, dropout=0.0):
        super(CrossAttentionPooling, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.query_num = query_num
        self.class_query = nn.Parameter(torch.randn(query_num, embed_dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=False
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(query_num * embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.class_query)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        b = x.shape[0]
        if x.dim() == 5:
            x = x.flatten(2)
        x = x.permute(2, 0, 1)  # [L, B, D]
        query = self.class_query.unsqueeze(1).repeat(1, b, 1)  # [Q, B, D]
        attended, _ = self.cross_attention(query=query, key=x, value=x)
        attended = self.norm(attended)
        attended = self.dropout(attended)
        attended_permuted = attended.permute(1, 0, 2)  # [B, Q, D]
        attended_flatten = attended_permuted.flatten(1)  # [B, Q*D]
        return self.classifier(attended_flatten)


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, query_num, num_classes, dropout=0.0,
                 use_cross_attention=True, num_heads=4):
        super(ClassificationHead, self).__init__()
        if use_cross_attention:
            self.pooling = CrossAttentionPooling(
                embed_dim, query_num, num_classes, num_heads, dropout
            )
        else:
            self.pooling = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(1),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes)
            )

    def forward(self, x):
        return self.pooling(x)

# ---------------------------------------------------------------------
# 3. 多任务模型 + 模式内置冻结逻辑
# ---------------------------------------------------------------------

class ResNet_MTL_nnUNet(nn.Module):
    """
    task_mode:
      - 'both'     : seg + cls 都训练
      - 'seg_only' : 只训练 segmentation，cls 头前向但不产生梯度
      - 'cls_only' : 只训练 classification，encoder + decoder + seg head 不产生梯度，
                     且 cls 分支梯度不会回流到 encoder
    """
    def __init__(
        self,
        # --- 必需参数 (无默认值) ---
        input_channels: int,
        n_stages: int,
        features_per_stage: Tuple[int, ...],
        conv_op: Type[nn.Module],
        kernel_sizes: Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
        strides: Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
        n_blocks_per_stage: Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
        num_classes: int,  # segmentation 类别数
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]],

        # classification 类别数
        cls_num_classes: int,

        # --- 可选参数 ---
        conv_bias: bool = False,
        norm_op: Type[nn.Module] = nn.InstanceNorm3d,
        norm_op_kwargs: dict = None,
        dropout_op: Type[nn.Module] = nn.Dropout3d,
        dropout_op_kwargs: dict = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = True,

        # ResNet 相关
        block: Type[nn.Module] = BasicBlockD,
        stem_channels: int = None,

        # 分类头相关（部分暂时不用）
        cls_query_num: int = 16,
        cls_dropout: float = 0.0,
        use_cross_attention: bool = False,
        cls_num_heads: int = 4,

        # 任务模式
        task_mode: str = 'cls_only',
    ):
        super().__init__()

        # 记录任务模式
        assert task_mode in ('both', 'seg_only', 'cls_only')
        self.task_mode = task_mode

        # 规范 nnU-Net 参数
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'inplace': True}

        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        self.n_stages = n_stages

        # ---------------------------------
        # 1. Encoder (ResNet)
        # ---------------------------------
        self.conv_encoder_blocks = nn.ModuleList()
        if stem_channels is None:
            stem_channels = features_per_stage[0]

        # stem
        self.conv_encoder_blocks.append(
            StackedResidualBlocks(
                n_blocks_per_stage[0], conv_op, input_channels, stem_channels,
                kernel_sizes[0], strides[0], conv_bias,
                norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                nonlin, nonlin_kwargs, block=block
            )
        )
        # 后续 stage
        for s in range(1, n_stages):
            self.conv_encoder_blocks.append(
                StackedResidualBlocks(
                    n_blocks_per_stage[s], conv_op,
                    features_per_stage[s - 1], features_per_stage[s],
                    kernel_sizes[s], strides[s], conv_bias,
                    norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                    nonlin, nonlin_kwargs, block=block
                )
            )

        # ---------------------------------
        # 2. Segmentation Decoder
        # ---------------------------------
        self.transpconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        transpconv_op = get_matching_convtransp(conv_op)

        for s in range(n_stages - 1, 0, -1):
            self.transpconvs.append(
                transpconv_op(
                    features_per_stage[s], features_per_stage[s - 1],
                    strides[s], strides[s], bias=conv_bias
                )
            )
            decoder_input_features = 2 * features_per_stage[s - 1]
            self.decoder_blocks.append(
                StackedConvBlocks(
                    n_conv_per_stage_decoder[s - 1], conv_op,
                    decoder_input_features, features_per_stage[s - 1],
                    kernel_sizes[s - 1], 1, conv_bias,
                    norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                    nonlin, nonlin_kwargs
                )
            )
            self.seg_layers.append(
                conv_op(features_per_stage[s - 1], num_classes, 1, 1, 0, bias=True)
            )

        # ---------------------------------
        # 3. Classification 分支 (bottleneck -> adapter -> GAP -> MLP)
        # ---------------------------------
        bottleneck_channels = features_per_stage[-1]
        total_cls_input_dim = sum(features_per_stage)  # 这里我们用 adapter 把 bottleneck -> 1120

        self.cls_adapter = nn.Sequential(
            StackedConvBlocks(
                2, conv_op, bottleneck_channels, bottleneck_channels,
                kernel_sizes[-1], 1, conv_bias, norm_op, norm_op_kwargs,
                dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
            ),
            conv_op(bottleneck_channels, total_cls_input_dim, 1, 1, 0, bias=True)
        )

        self.classification_head = nn.Sequential(
            nn.Linear(total_cls_input_dim, total_cls_input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(cls_dropout),
            nn.Linear(total_cls_input_dim // 2, cls_num_classes)
        )

    # ======== 一些小工具函数 ========

    def set_task_mode(self, mode: str):
        """外部可以随时切模式，例如 net.set_task_mode('seg_only')"""
        assert mode in ('both', 'seg_only', 'cls_only')
        self.task_mode = mode

    # segmentation 分支前向（不关心梯度，在外层控制）
    def _forward_seg_branch(self, bottleneck_features, skips):
        seg_outputs = []
        x_dec = bottleneck_features
        for i in range(len(self.decoder_blocks)):
            skip_connection = skips[-(i + 2)]
            x_dec = self.transpconvs[i](x_dec)
            x_dec = torch.cat((x_dec, skip_connection), dim=1)
            x_dec = self.decoder_blocks[i](x_dec)
            seg_outputs.append(self.seg_layers[i](x_dec))

        if self.deep_supervision:
            seg_output_return = seg_outputs[::-1]
        else:
            seg_output_return = seg_outputs[-1]
        return seg_output_return

    # classification 分支前向（输入是某个 feature map）
    def _forward_cls_branch(self, bottleneck_features):
        adapted = self.cls_adapter(bottleneck_features)            # [B, 1120, D, H, W]
        cls_vec = adapted.mean(dim=(2, 3, 4))                      # [B, 1120]
        out = self.classification_head(cls_vec)                    # [B, cls_num_classes]
        return out

    # ======== 主 forward，内置模式控制梯度 ========

    def forward(self, x: torch.Tensor,
                task_mode: str = None) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:
        """
        x: [B, C, D, H, W]
        task_mode: 可选覆盖 self.task_mode；不传就用当前对象的 task_mode
        """
        mode = task_mode if task_mode is not None else self.task_mode
        assert mode in ('both', 'seg_only', 'cls_only')

        # 1. Encoder
        skips: List[torch.Tensor] = []
        x_enc = x
        for i in range(self.n_stages):
            x_enc = self.conv_encoder_blocks[i](x_enc)
            skips.append(x_enc)
        bottleneck_features = skips[-1]

        # 2. Segmentation 路径
        if mode == 'cls_only':
            # 只训分类：seg 路径不参与梯度
            with torch.no_grad():
                seg_output = self._forward_seg_branch(bottleneck_features, skips)
        else:
            # both / seg_only：seg 正常产生梯度
            seg_output = self._forward_seg_branch(bottleneck_features, skips)

        # 3. Classification 路径
        if mode == 'seg_only':
            # 只训 seg：cls 分支只做前向，无梯度
            with torch.no_grad():
                cls_output = self._forward_cls_branch(bottleneck_features)
        elif mode == 'cls_only':
            # 只训 cls：不让梯度回到 encoder（detach），只更新 cls_adapter + classification_head
            bottleneck_detached = bottleneck_features.detach()
            cls_output = self._forward_cls_branch(bottleneck_detached)
        else:
            # both：seg + cls 都回传到 encoder
            cls_output = self._forward_cls_branch(bottleneck_features)

        return seg_output, cls_output


# ---------------------------------------------------------------------
# 简单自测
# ---------------------------------------------------------------------
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seg_classes = 14
    cls_classes = 3

    unet_config = {
        "input_channels": 1,
        "n_stages": 6,
        "features_per_stage": (32, 64, 128, 256, 320, 320),
        "conv_op": nn.Conv3d,
        "kernel_sizes": ((3, 3, 3),) * 6,
        "strides": ((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        "n_blocks_per_stage": (2, 2, 2, 2, 2, 2),
        "num_classes": seg_classes,
        "n_conv_per_stage_decoder": (2, 2, 2, 2, 2),
        "conv_bias": True,
        "norm_op": nn.InstanceNorm3d,
        "norm_op_kwargs": {"eps": 1e-05, "affine": True},
        "dropout_op": None,
        "dropout_op_kwargs": None,
        "nonlin": nn.LeakyReLU,
        "nonlin_kwargs": {"inplace": True},
        "deep_supervision": True,
        "block": BasicBlockD,
    }

    model = ResNet_MTL_nnUNet(
        **unet_config,
        cls_num_classes=cls_classes,
        cls_query_num=16,
        cls_dropout=0.1,
        use_cross_attention=False,
        task_mode='both',
    ).to(device)

    dummy_input = torch.randn(2, 1, 128, 128, 128).to(device)

    with torch.no_grad():
        seg_output, class_output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Classification output shape: {class_output.shape}")
    print("Segmentation outputs (Deep Supervision):")
    if isinstance(seg_output, list):
        for i, out in enumerate(seg_output):
            print(f"  Level {i}: {out.shape}")
    else:
        print(f"  Output shape: {seg_output.shape}")
