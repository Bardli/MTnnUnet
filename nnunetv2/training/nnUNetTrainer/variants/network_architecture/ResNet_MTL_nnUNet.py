import torch
import torch.nn as nn
from typing import Tuple, Union, List, Type
import os
import torch.nn.functional as F
# ---------------------------------------------------------------------
# 1. 导入 nnU-Net v2 的核心构建模块
# ---------------------------------------------------------------------
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks, BasicBlockD
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp, maybe_convert_scalar_to_list

    
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
        if x.dim() == 5: x = x.flatten(2)
        x = x.permute(2, 0, 1) # [L, B, D]
        query = self.class_query.unsqueeze(1).repeat(1, b, 1) # [query_num, B, D]
        attended, _ = self.cross_attention(query=query, key=x, value=x)
        attended = self.norm(attended)
        attended = self.dropout(attended)
        attended_permuted = attended.permute(1, 0, 2) # [B, query_num, D]
        attended_flatten = attended_permuted.flatten(1) # [B, query_num * D]
        return self.classifier(attended_flatten)

class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, query_num, num_classes, dropout=0.0, use_cross_attention=True, num_heads=4):
        super(ClassificationHead, self).__init__()
        if use_cross_attention:
            self.pooling = CrossAttentionPooling(
                embed_dim, query_num, num_classes, num_heads, dropout
            )
        else:
            self.pooling = nn.Sequential(
                nn.AdaptiveAvgPool3d(1), nn.Flatten(1), nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes)
            )
    def forward(self, x):
        return self.pooling(x)

# ---------------------------------------------------------------------
# 3. 修正后的多任务模型 (继承自 nn.Module)
# ---------------------------------------------------------------------

class ResNet_MTL_nnUNet(nn.Module):
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
        num_classes: int,  # 分割任务的类别数
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]],

        # 新增的必需参数 (分类类别数)
        cls_num_classes: int,

        # --- 可选参数 (有默认值) ---
        conv_bias: bool = False,
        norm_op: Type[nn.Module] = nn.InstanceNorm3d,
        norm_op_kwargs: dict = None,
        dropout_op: Type[nn.Module] = nn.Dropout3d,
        dropout_op_kwargs: dict = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = True,

        # --- ResNet 特定可选参数 ---
        block: Type[nn.Module] = BasicBlockD,
        stem_channels: int = None,

        # --- 分类头可选参数 ---
        cls_query_num: int = 16,     # 现在不用了，保留参数不影响
        cls_dropout: float = 0.0,
        use_cross_attention: bool = True,  # 现在不用了，保留参数不影响
        cls_num_heads: int = 4
    ):
        super().__init__()

        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'inplace': True}

        self.deep_supervision = deep_supervision
        self.num_classes = num_classes  # 分割类别数
        self.n_stages = n_stages

        # ---------------------------------
        # 1. 构建共享 3D ResNet 编码器
        # ---------------------------------
        self.conv_encoder_blocks = nn.ModuleList()
        if stem_channels is None:
            stem_channels = features_per_stage[0]

        self.conv_encoder_blocks.append(
            StackedResidualBlocks(
                n_blocks_per_stage[0], conv_op, input_channels, stem_channels, kernel_sizes[0], strides[0],
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                nonlin, nonlin_kwargs, block=block
            )
        )
        for s in range(1, n_stages):
            self.conv_encoder_blocks.append(
                StackedResidualBlocks(
                    n_blocks_per_stage[s], conv_op, features_per_stage[s - 1], features_per_stage[s],
                    kernel_sizes[s], strides[s], conv_bias, norm_op, norm_op_kwargs,
                    dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, block=block
                )
            )

        # ---------------------------------
        # 2. 构建分割解码器 (Seg Head)
        # ---------------------------------
        self.transpconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        transpconv_op = get_matching_convtransp(conv_op)

        for s in range(n_stages - 1, 0, -1):
            self.transpconvs.append(
                transpconv_op(
                    features_per_stage[s], features_per_stage[s - 1], strides[s], strides[s],
                    bias=conv_bias
                )
            )
            decoder_input_features = 2 * features_per_stage[s - 1]
            self.decoder_blocks.append(
                StackedConvBlocks(
                    n_conv_per_stage_decoder[s - 1], conv_op, decoder_input_features, features_per_stage[s - 1],
                    kernel_sizes[s - 1], 1, conv_bias, norm_op, norm_op_kwargs,
                    dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
                )
            )
            self.seg_layers.append(
                conv_op(features_per_stage[s - 1], num_classes, 1, 1, 0, bias=True)
            )

        # ---------------------------------
        # 3. 分类头：基于 seg soft mask 的 ROI pooling + 3 层 MLP
        # ---------------------------------
        bottleneck_channels = features_per_stage[-1]
        # 在 bottleneck 上再堆 2 个 residual block，专门给 cls 用
        self.cls_res_block = StackedResidualBlocks(
            2,                    # n_blocks
            conv_op,
            bottleneck_channels,  # in_channels
            bottleneck_channels,  # out_channels
            kernel_sizes[-1],     # kernel_size
            1,                    # stride=1（不改变空间尺寸）
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            block=block
        )


        self.classification_head = nn.Linear(bottleneck_channels, cls_num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:

        skips = []

        # ---------------------------------
        # 1. 编码器
        # ---------------------------------
        x_enc = x
        for i in range(self.n_stages):
            x_enc = self.conv_encoder_blocks[i](x_enc)
            skips.append(x_enc)

        # bottleneck 特征 (最高语义层，用来做 ROI pooling)
        bottleneck_features = skips[-1]  # [B, C_bottleneck, D_b, H_b, W_b]

        # ---------------------------------
        # 2. 分割解码器 (Seg Decoder)
        # ---------------------------------
        # segmentation 只用 skips_detach，不往 encoder 回传梯度
        skips_detach = [s.detach() for s in skips]
        seg_outputs = []
        x_dec = bottleneck_features.detach()

         # 2. 分割解码器 (Seg Decoder)
        seg_outputs = []
        for i in range(len(self.decoder_blocks)):
            skip_connection = skips_detach[-(i + 2)]
            x_dec = self.transpconvs[i](x_dec)
            x_dec = torch.cat((x_dec, skip_connection), dim=1)
            x_dec = self.decoder_blocks[i](x_dec)
            seg_outputs.append(self.seg_layers[i](x_dec))

        # 3. 分类分支：cls_res_block + GAP + Linear
        #    只用 bottleneck_features，不被 seg 污染梯度
        cls_feat = self.cls_res_block(bottleneck_features)         # [B, Cb, Db, Hb, Wb]
        gap = cls_feat.mean(dim=(2, 3, 4))                         # [B, Cb]
        class_output = self.classification_head(gap)               # [B, cls_num_classes]

        # 4. deep supervision 格式保持不变
        if self.deep_supervision:
            seg_output_return = seg_outputs[::-1]  # 从高分辨率到低分辨率
        else:
            seg_output_return = seg_outputs[-1]

        return seg_output_return, class_output

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. 定义 nnU-Net v2 的标准 "3d_fullres" ResNet 配置 ---
    
    # 任务1：分割 (假设14个类别，包括背景)
    seg_classes = 14 
    
    # 任务2：分类 (假设3个类别)
    cls_classes = 3 
    
    # nnU-Net v2 典型配置
    unet_config = {
        "input_channels": 1,
        "n_stages": 6,
        "features_per_stage": (32, 64, 128, 256, 320, 320),
        "conv_op": nn.Conv3d,
        "kernel_sizes": ((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
        "strides": ((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        "n_blocks_per_stage": (2, 2, 2, 2, 2, 2),
        "num_classes": seg_classes, # 传入分割类别数
        "n_conv_per_stage_decoder": (2, 2, 2, 2, 2),
        "conv_bias": True,
        "norm_op": nn.InstanceNorm3d,
        "norm_op_kwargs": {"eps": 1e-05, "affine": True},
        "dropout_op": None,
        "dropout_op_kwargs": None,
        "nonlin": nn.LeakyReLU,
        "nonlin_kwargs": {"inplace": True},
        "deep_supervision": True,
        "block": BasicBlockD, # 这使其成为 ResNet 编码器
    }

    # --- 2. 实例化我们的多任务模型 ---
    # *******************************************************
    # *** 唯一的更改：使用我们新的、从 nn.Module 继承的类 ***
    # *******************************************************
    model = ResNet_MTL_nnUNet(
        **unet_config, # 传入所有 nnU-Net 标准参数
        
        # 传入分类头的特定参数
        cls_num_classes=cls_classes,
        cls_query_num=16,          # <-- 这个参数现在没用了，但保留它不会出错
        cls_dropout=0.1,
        use_cross_attention=True   # <-- 这个参数现在也没用了，但保留它不会出错
    ).to(device)

    # --- 3. 测试前向传播 ---
    
    dummy_input = torch.randn(2, 1, 128, 128, 128).to(device) 

    with torch.no_grad():
        seg_output, class_output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print("-" * 30)
    print(f"Classification output shape: {class_output.shape}")
    print(f"Expected: (2, {cls_classes})")
    print("-" * 30)
    
    print("Segmentation outputs (Deep Supervision):")
    if isinstance(seg_output, list):
        for i, out in enumerate(seg_output):
            print(f"   Level {i} (High-res to Low-res): {out.shape}")
    else:
        print(f"   Output shape: {seg_output.shape}")
        
    # 检查分类头的输入维度是否正确
    # (32+64+128+256+320+320 = 1120)
    expected_dim = sum(unet_config['features_per_stage'])
    print("-" * 30)
    print(f"Classification head MLP input dim: {model.classification_head[0].in_features}")
    print(f"Expected MLP input dim:        {expected_dim}")
    assert model.classification_head[0].in_features == expected_dim
    print("Shape check PASSED.")