import torch
import torch.nn as nn
from typing import Tuple, Union, List, Type
import os

# ---------------------------------------------------------------------
# 1. 导入 nnU-Net v2 的核心构建模块
# ---------------------------------------------------------------------
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks, BasicBlockD
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp, maybe_convert_scalar_to_list
# --- GradNorm START: 添加新 imports ---
from torch.autograd import grad
import torch.nn.functional as F
# ---------------------------------------------------------------------
# 2. 复用 ClassificationHead 和 CrossAttentionPooling
# (这部分代码无需修改)
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
        num_classes: int, # 分割任务的类别数
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
        
        # --- ★★★ 修正点 ★★★ ---
        # 新增的必需参数 (分类类别数)
        cls_num_classes: int, 
        # ------------------------

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
        cls_query_num: int = 16,
        cls_dropout: float = 0.0,
        use_cross_attention: bool = True,
        cls_num_heads: int = 4
    ):
        super().__init__()
        # --- 检查和格式化 nnU-Net 参数 ---
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'inplace': True}
            
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes # 分割类别数
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
        # 3. ★★★ 构建新的聚合分类头 (Cls Head) ★★★
        # ---------------------------------
        
        # 瓶颈层的特征数
        bottleneck_features_dim = features_per_stage[-1]
        
        # 总特征维度 = 编码器所有阶段 + 瓶颈层 的特征总和
        total_cls_input_dim = sum(features_per_stage)

        # --- 新增：为分类任务添加一个“适配器” ---
        # 这个模块将瓶颈特征图转换为分类特征图
        self.cls_adapter = nn.Sequential(
            StackedConvBlocks(
                2, conv_op, bottleneck_features_dim, bottleneck_features_dim,
                kernel_sizes[-1], 1, conv_bias, norm_op, norm_op_kwargs,
                dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
            ),
            # 你可以选择在这里使用 1x1x1 卷积来改变维度，
            # 但保持维度一致通常是安全的。
            conv_op(bottleneck_features_dim, total_cls_input_dim, 1, 1, 0, bias=True)
        )
        

        # 你的 classification_head 定义保持不变
        # (无论是使用 CrossAttention 还是简单的 MLP)
        self.classification_head = nn.Sequential(
        nn.Linear(total_cls_input_dim, total_cls_input_dim // 2),
        nn.ReLU(inplace=True),
        nn.Dropout(cls_dropout),
        nn.Linear(total_cls_input_dim // 2, cls_num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:
        
        skips = []
        cls_feature_list = [] # 用于 PANDA/iAorta 策略的特征列表
        
        # ---------------------------------
        # 1. & 2. 编码器 + 瓶颈层
        # ---------------------------------
        # self.conv_encoder_blocks 列表包含了 stem 和所有 encoder 阶段
        # (共 n_stages 个块)
        x_enc = x
        
        # 迭代 n_stages 次 (例如 6 次)
        # self.n_stages 是在 __init__ 中定义的
        for i in range(self.n_stages): 
            x_enc = self.conv_encoder_blocks[i](x_enc)
            skips.append(x_enc)
            cls_feature_list.append(x_enc) 
            # cls_feature_list 现在包含了 stem, stage 1, ..., stage 5 (瓶颈层)
            # 的所有 (n_stages) 个特征图
        
        # 瓶颈层特征已经是 skips 列表的最后一项
        bottleneck_features = skips[-1] 
        # (不需要再向 cls_feature_list 添加任何东西)

        # ---------------------------------
        # 3. 分割解码器 (Seg Decoder)
        # ---------------------------------
        seg_outputs = []
        x_dec = bottleneck_features
        
        # 迭代 (n_stages - 1) 次 (例如 5 次)
        for i in range(len(self.decoder_blocks)):
            # 编码器有 n_stages 个输出 (skips 列表)
            # 解码器有 (n_stages - 1) 个块
            # 第 i 个解码器块 (i=0..n_stages-2)
            # 需要连接第 (n_stages - 2 - i) 个 skip
            # 这等于 skips[-(i + 2)]
            skip_connection = skips[-(i + 2)] 
            x_dec = self.transpconvs[i](x_dec)
            x_dec = torch.cat((x_dec, skip_connection), dim=1)
            x_dec = self.decoder_blocks[i](x_dec)
            
            seg_outputs.append(self.seg_layers[i](x_dec))

        # ---------------------------------
        # 4. ★★★ 完成分类头的计算 ★★★
        # ---------------------------------
        # (d) cls_feature_list 已经包含了所有编码器/瓶颈层的特征图
        #    它的长度应该等于 n_stages
        pooled_features = [f.mean(dim=[-1, -2, -3]) for f in cls_feature_list]
        
        # (e) 拼接所有尺度的特征向量
        combined_vector = torch.cat(pooled_features, dim=1)

        # 添加 cls_adapter 调用（
        adapted_feature = self.cls_adapter(cls_feature_list[-1])  # <--- 🔥 用瓶颈层特征适配
        class_input = adapted_feature.mean(dim=[-1, -2, -3])   # shape: [B, 1120]
        # (f) 得到最终分类输出
        #     self.classification_head 是你在 __init__ 中定义的 MLP
        # class_output = self.classification_head(combined_vector)
        class_output = self.classification_head(class_input)

        # ---------------------------------
        # 5. 格式化分割输出 (保持不变)
        # ---------------------------------
        if self.deep_supervision:
            seg_output_return = seg_outputs[::-1] # 反转列表以匹配 nnU-Net 深度监督
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
        cls_query_num=16,          # 使用 16 个 query "探针"
        cls_dropout=0.1,
        use_cross_attention=True
    ).to(device)

    # --- 3. 测试前向传播 ---
    
    dummy_input = torch.randn(2, 1, 128, 128, 128).to(device) 

    with torch.no_grad():
        seg_output, class_output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print("-" * 30)
    print(f"Classification output shape: {class_output.shape}")
    print("-" * 30)
    
    print("Segmentation outputs (Deep Supervision):")
    if isinstance(seg_output, list):
        for i, out in enumerate(seg_output):
            print(f"  Level {i} (High-res to Low-res): {out.shape}")
    else:
        print(f"  Output shape: {seg_output.shape}")