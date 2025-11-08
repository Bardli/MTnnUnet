import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskResidualUNet(nn.Module):
    """
    Multi-task 3D U-Net with a residual encoder and dual heads for segmentation and classification.
    
    - Encoder: nnU-Net ResidualEncoderUNet architecture (stacked residual blocks with down-sampling).
    - Segmentation decoder: Standard U-Net decoder (transpose conv + convolution blocks) with optional deep supervision.
    - Classification head: Uses multi-scale encoder features + segmentation mask information with attention fusion, 
      followed by two fully connected layers for three-class classification.
    
    Parameters:
        input_channels (int): Number of input image channels.
        num_classes (int): Number of segmentation output classes (including background).
        cls_num_classes (int): Number of classes for classification output.
        features_per_stage (tuple of int): Number of feature channels at each encoder stage (length = n_stages).
        n_blocks_per_stage (int or tuple): Number of residual blocks per encoder stage. If int, the same number is used for all stages.
        kernel_sizes (int or tuple of tuples): Convolution kernel sizes for each stage. Can be an int (same for all) or a tuple of size n_stages.
        strides (int or tuple of tuples): Convolution strides for each stage (defines downsampling). Can be int or tuple of size n_stages.
        n_conv_per_stage_decoder (int or tuple): Number of conv layers in each decoder block (after concatenation with skip). If int, used for all decoder stages.
        deep_supervision (bool): If True, segmentation output is a list of outputs at each decoder stage (from lowest resolution to highest). If False, only final segmentation is returned.
        cls_dropout (float): Dropout rate in the classification head (for the first FC layer).
        task_mode (str): 'both', 'seg_only', or 'cls_only'. Determines which branches compute gradients during training.
    """
    def __init__(self, 
                 input_channels: int,
                 num_classes: int,
                 cls_num_classes: int,
                 features_per_stage: tuple,
                 n_blocks_per_stage: int or tuple,
                 kernel_sizes: int or tuple,
                 strides: int or tuple,
                 n_conv_per_stage_decoder: int or tuple,
                 deep_supervision: bool = True,
                 cls_dropout: float = 0.5,
                 task_mode: str = 'both'):
        super().__init__()
        assert task_mode in ('both', 'seg_only', 'cls_only'), "Invalid task_mode"
        self.task_mode = task_mode
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        
        # If scalar inputs are provided for per-stage params, expand them to tuples
        self.n_stages = len(features_per_stage)
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = tuple([n_blocks_per_stage] * self.n_stages)
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = tuple([n_conv_per_stage_decoder] * (self.n_stages - 1))
        # Ensure kernel_sizes and strides are in tuple-of-tuples format
        def ensure_tuple_of_tuples(param, name):
            if isinstance(param, int):
                return tuple([(param,)*3] * self.n_stages)
            if isinstance(param, tuple) and len(param) == self.n_stages and not isinstance(param[0], tuple):
                # If given a tuple of length n_stages with ints, convert each int to a 3-tuple
                return tuple([ (p,)*3 for p in param ])
            if isinstance(param, tuple) and len(param) == self.n_stages:
                return tuple(param)
            raise ValueError(f"{name} must be int or tuple of length {self.n_stages}")
        kernel_sizes = ensure_tuple_of_tuples(kernel_sizes, "kernel_sizes")
        strides = ensure_tuple_of_tuples(strides, "strides")
        
        # Define a basic 3D residual block (two conv layers + skip connection)
        class ResidualBlock3d(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride):
                super().__init__()
                padding = tuple(k//2 for k in kernel_size)
                self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
                self.norm1 = nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True)
                self.act1 = nn.LeakyReLU(inplace=True)
                self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride=(1,1,1), padding=padding, bias=False)
                self.norm2 = nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True)
                self.act2 = nn.LeakyReLU(inplace=True)
                # If channels or spatial size change, define a projection for the residual
                if in_channels != out_channels or any(s > 1 for s in (stride if isinstance(stride, tuple) else (stride,)*3)):
                    self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,1), stride=stride, bias=False)
                    self.res_norm = nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True)
                else:
                    self.res_conv = None
                    self.res_norm = None
            def forward(self, x):
                identity = x
                out = self.conv1(x)
                out = self.norm1(out)
                out = self.act1(out)
                out = self.conv2(out)
                out = self.norm2(out)
                # Residual connection
                if self.res_conv is not None:
                    identity = self.res_conv(identity)
                    identity = self.res_norm(identity)
                out = out + identity
                out = self.act2(out)
                return out
        
        # -------------------------
        # Encoder (Residual blocks)
        # -------------------------
        self.conv_encoder_blocks = nn.ModuleList()
        for stage in range(self.n_stages):
            in_ch = input_channels if stage == 0 else features_per_stage[stage-1]
            out_ch = features_per_stage[stage]
            n_blocks = n_blocks_per_stage[stage]
            ks = kernel_sizes[stage]
            st = strides[stage]
            # Build a stage: a stack of residual blocks (the first block handles downsampling via stride if st != (1,1,1))
            blocks = []
            for block_idx in range(n_blocks):
                # First block in stage uses the given stride, subsequent ones use stride 1
                stride = st if block_idx == 0 else (1,1,1)
                in_channels = in_ch if block_idx == 0 else out_ch
                blocks.append(ResidualBlock3d(in_channels, out_ch, ks, stride))
            self.conv_encoder_blocks.append(nn.Sequential(*blocks))
        
        # -------------------------
        # Decoder (Transpose conv + conv blocks for segmentation)
        # -------------------------
        self.transpconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        # Create decoder stages from deepest to second stage (since U-Net decoder goes from bottleneck up to stage1)
        for stage in range(self.n_stages-1, 0, -1):
            # Transposed conv to upsample from stage to stage-1
            self.transpconvs.append(nn.ConvTranspose3d(
                features_per_stage[stage], features_per_stage[stage-1],
                kernel_size=strides[stage], stride=strides[stage]
            ))
            # After upsampling, concatenation with corresponding encoder skip will double channels
            dec_in_channels = features_per_stage[stage-1] * 2
            dec_out_channels = features_per_stage[stage-1]
            n_conv = n_conv_per_stage_decoder[stage-1]
            # Decoder convolutional block (without further downsampling, stride=1)
            convs = []
            for i in range(n_conv):
                in_ch = dec_in_channels if i == 0 else dec_out_channels
                out_ch = dec_out_channels
                convs.append(nn.Conv3d(in_ch, out_ch, kernel_size=kernel_sizes[stage-1], padding=tuple(k//2 for k in kernel_sizes[stage-1]), bias=False))
                convs.append(nn.InstanceNorm3d(out_ch, eps=1e-5, affine=True))
                convs.append(nn.LeakyReLU(inplace=True))
            self.decoder_blocks.append(nn.Sequential(*convs))
            # Segmentation output layer for this decoder stage
            self.seg_layers.append(nn.Conv3d(dec_out_channels, num_classes, kernel_size=1))
        
        # -------------------------
        # Classification branch
        # -------------------------
        # Determine channels from last two encoder feature maps
        enc_ch_last = features_per_stage[-1]          # channels of bottleneck
        enc_ch_second_last = features_per_stage[-2]   # channels of second-last encoder layer
        classif_feat_dim = enc_ch_last + enc_ch_second_last  # dimension of concatenated multi-scale features
        # Attention fusion module: learns to weight global vs region features (2 sources)
        self.attention_fuse = nn.Sequential(
            nn.Linear(classif_feat_dim * 2, 2),       # input is concatenated [global, region] features
            nn.Softmax(dim=1)                        # output 2 weights (for global vs region)
        )
        # Two-layer classification head (fully connected)
        hidden_dim = max(1, classif_feat_dim // 2)    # hidden layer size (half of feature dim, at least 1)
        self.classifier_fc = nn.Sequential(
            nn.Linear(classif_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cls_dropout),
            nn.Linear(hidden_dim, cls_num_classes)
        )
    
    def forward(self, x: torch.Tensor, task_mode: str = None):
        """
        Forward pass. Depending on task_mode, returns:
            - (segmentation_output, classification_output)
        Segmentation output is either a list of tensors (if deep_supervision=True) or a single tensor.
        Classification output is a tensor of shape (B, cls_num_classes).
        """
        mode = task_mode if task_mode is not None else self.task_mode
        assert mode in ('both', 'seg_only', 'cls_only'), "Invalid task_mode"
        
        # -------------------------
        # Encoder forward
        # -------------------------
        skips = []  # store outputs at each stage for skip connections
        out = x
        for enc_block in self.conv_encoder_blocks:
            out = enc_block(out)
            skips.append(out)
        # Now, skips[-1] is the bottleneck feature map, skips[-2] is second-last encoder feature map
        bottleneck_feat = skips[-1]             # shape [B, enc_ch_last, D_b, H_b, W_b]
        second_last_feat = skips[-2]            # shape [B, enc_ch_second_last, D_s, H_s, W_s]
        
        # -------------------------
        # Segmentation decoder forward
        # -------------------------
        if mode == 'cls_only':
            # If only classification, run segmentation decoder in inference mode (no grad)
            with torch.no_grad():
                seg_outputs = self._forward_seg_decoder(bottleneck_feat, skips)
        else:
            seg_outputs = self._forward_seg_decoder(bottleneck_feat, skips)
        
        # Prepare the final segmentation output (for returning and for classification mask usage)
        if isinstance(seg_outputs, list):
            # If deep supervision, final output is the last element (highest resolution)
            seg_logits = seg_outputs[-1]
        else:
            seg_logits = seg_outputs  # no deep supervision, seg_outputs is final logits tensor
        
        # -------------------------
        # Classification branch forward
        # -------------------------
        if mode == 'seg_only':
            # If only segmentation, compute classification logits without grad (do not update classifier)
            with torch.no_grad():
                cls_output = self._forward_cls_branch(bottleneck_feat, second_last_feat, seg_logits)
        elif mode == 'cls_only':
            # If only classification, freeze encoder features (no grad through encoder) for classification
            cls_output = self._forward_cls_branch(bottleneck_feat.detach(), second_last_feat.detach(), seg_logits.detach())
        else:
            # Both tasks: full grad through encoder, decoder, and classifier
            cls_output = self._forward_cls_branch(bottleneck_feat, second_last_feat, seg_logits)
        
        return seg_outputs, cls_output
    
    def _forward_seg_decoder(self, bottleneck_feat: torch.Tensor, skips: list):
        """Forward pass through the U-Net decoder for segmentation."""
        seg_outputs = []
        x = bottleneck_feat
        # Iterate through decoder stages (note: self.decoder_blocks and self.transpconvs are in reverse order of encoder)
        for i in range(len(self.decoder_blocks)):
            # Corresponding skip connection from encoder (in reverse order: skip[-2] for first decoder block, etc.)
            # skips list has length n_stages, and we have n_stages-1 decoder blocks. For decoder index i:
            # use skip index = -(i+2) (e.g., i=0 -> skip[-2], i=1 -> skip[-3], ..., i=last -> skip[-(n_stages)])
            skip_feat = skips[-(i+2)]
            # Transpose convolution (upsample)
            x = self.transpconvs[i](x)
            # Concatenate skip connection
            x = torch.cat((x, skip_feat), dim=1)
            # Decoder convolution block
            x = self.decoder_blocks[i](x)
            # Segmentation output at this scale
            seg_logit = self.seg_layers[i](x)
            seg_outputs.append(seg_logit)
        # Return segmentation outputs (list or final tensor depending on deep supervision)
        if self.deep_supervision:
            # If deep supervision, return list of outputs (from lowest resolution to highest resolution)
            return seg_outputs[::-1]  # reverse to have highest resolution (final) first if desired
        else:
            # Only final output
            return seg_outputs[-1]
    
    def _forward_cls_branch(self, bottleneck_feat: torch.Tensor, second_last_feat: torch.Tensor, seg_logits: torch.Tensor):
        """Forward pass for classification branch. Computes multi-scale features, fuses with segmentation mask, and applies FC layers."""
        # Global average pooling on multi-scale encoder features
        f_global_bottleneck = bottleneck_feat.mean(dim=(2, 3, 4))     # [B, enc_ch_last]
        f_global_second = second_last_feat.mean(dim=(2, 3, 4))        # [B, enc_ch_second_last]
        f_global = torch.cat([f_global_bottleneck, f_global_second], dim=1)  # [B, classif_feat_dim]
        
        # Derive segmentation probability mask for lesion/foreground
        if self.num_classes > 1:
            # If multiple segmentation classes, assume channel 0 is background – use probability of foreground class 1 (or combined foreground)
            seg_prob = F.softmax(seg_logits, dim=1)                   # [B, num_classes, D, H, W]
            # Use class 1 as foreground mask (if multiple foreground classes exist, this can be adjusted or combined as needed)
            if seg_prob.shape[1] > 1:
                seg_mask_fullres = seg_prob[:, 1]                     # [B, D, H, W] probability of class 1
            else:
                seg_mask_fullres = seg_prob[:, 0]                     # [B, D, H, W] (edge case: if only one channel without explicit background)
            seg_mask_fullres = seg_mask_fullres.unsqueeze(1)          # [B, 1, D, H, W]
        else:
            # If num_classes == 1 (segmentation map is single-channel logits), apply sigmoid to get probability
            seg_mask_fullres = torch.sigmoid(seg_logits).unsqueeze(1) # [B, 1, D, H, W]
        
        # Downsample the segmentation mask to the spatial size of the encoder features
        # Bottleneck feature map size:
        D_b, H_b, W_b = bottleneck_feat.shape[2:]
        D_s, H_s, W_s = second_last_feat.shape[2:]
        mask_bottleneck = F.interpolate(seg_mask_fullres, size=(D_b, H_b, W_b), mode='trilinear', align_corners=False)
        mask_second = F.interpolate(seg_mask_fullres, size=(D_s, H_s, W_s), mode='trilinear', align_corners=False)
        
        # Masked global average pooling (focus on segmentation regions)
        f_region_bottleneck = (bottleneck_feat * mask_bottleneck).mean(dim=(2, 3, 4))  # [B, enc_ch_last]
        f_region_second = (second_last_feat * mask_second).mean(dim=(2, 3, 4))         # [B, enc_ch_second_last]
        f_region = torch.cat([f_region_bottleneck, f_region_second], dim=1)            # [B, classif_feat_dim]
        
        # Attention-based fusion of global vs region features
        # Concatenate global and region vectors and compute attention weights
        concat_features = torch.cat([f_global, f_region], dim=1)      # [B, 2*classif_feat_dim]
        weights = self.attention_fuse(concat_features)                # [B, 2], softmax gives [w_global, w_region] per sample
        w_global = weights[:, 0].unsqueeze(1)                         # [B, 1]
        w_region = weights[:, 1].unsqueeze(1)                         # [B, 1]
        # Fuse features as weighted sum
        fused_features = w_global * f_global + w_region * f_region    # [B, classif_feat_dim]
        
        # Final classification via two FC layers
        cls_logits = self.classifier_fc(fused_features)               # [B, cls_num_classes]
        return cls_logits
