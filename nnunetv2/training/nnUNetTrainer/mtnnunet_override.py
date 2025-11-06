"""
This file provides a set of overrides to enable multi‑task training within
the nnUNetv2 framework using a 3D ResNet‑M encoder and dual decoder
heads: one for segmentation and one for lesion subtype classification.

The supplied classes are designed to be drop‑in replacements for the
standard nnUNet v2 planner and trainer.  To use them you should point
``nnUNetv2_plan_and_preprocess`` at ``nnUNetPlannerResEncM_MTL`` and
``nnUNetv2_train`` at ``nnUNetTrainerResEncMMTL``.  For example:

    nnUNetv2_plan_and_preprocess -d 002 -pl nnUNetPlannerResEncM_MTL
    nnUNetv2_train 002 3d_fullres 4 \
        -p nnUNetResEncUNetMPlans_MTL -tr nnUNetTrainerResEncMMTL

The resulting network shares a common 3D ResNet‑M encoder between
segmentation and classification tasks.  The segmentation decoder is a
standard U‑Net style decoder with deep supervision support.  The
classification branch passes the deepest encoder feature map through
a small adapter and MLP to produce logits for three lesion subtypes
(0, 1, 2).  During training the total loss is a weighted sum of
segmentation (Dice + CrossEntropy) and classification (CrossEntropy)
losses.  Early epochs emphasise classification to encourage the
encoder to learn subtype‑specific features.

Notes
-----
* This override assumes your dataset is organised according to the
  accompanying task description: file names follow
  ``quiz_<subtype>_<caseid>_0000.nii.gz`` for images and
  ``quiz_<subtype>_<caseid>.nii.gz`` for masks.  The subtype index
  extracted from the filename (0, 1, 2) serves as the classification
  label.
* The custom trainer will parse these subtype labels automatically
  from the training keys and compute class weights to mitigate
  imbalance.  Make sure your preprocessed dataset preserves the
  original case identifiers (nnUNet does this by default).
* Wandb logging and other niceties from the base trainer are left
  intact.  Only the points necessary to insert classification
  behaviour have been touched.

"""

from __future__ import annotations
from time import time, sleep
import os
import pydoc
from typing import Tuple, Union, List, Type
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1. Multi‑task network with ResNet‑M encoder
#
# We reuse the ResNet_MTL_nnUNet implementation from the provided repository
# (see ``nnunetv2/training/nnUNetTrainer/variants/network_architecture/ResNet_MTL_nnUNet.py``).
# It accepts the same set of architectural arguments as the standard
# ResidualEncoderUNet but exposes two additional parameters:
#   cls_num_classes : number of lesion subtype classes
#   task_mode       : 'both', 'seg_only' or 'cls_only'
#
# To simplify integration with nnUNet planners we wrap the model in a thin
# adapter which exposes a matching interface.
# -----------------------------------------------------------------------------
class DummyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # nnUNet Trainer 只会改这个开关，不会真的用这个 decoder 做 forward
        self.deep_supervision = False

class ResNet_MTL_nnUNet(nn.Module):
    """
    Multi‑task U‑Net with a 3D ResNet‑M encoder.  The implementation is
    copied from the original project (see documentation for details).

    Parameters
    ----------
    input_channels : int
        Number of input image channels.
    n_stages : int
        Number of resolution stages in the encoder/decoder.  Must match the
        number of entries in ``features_per_stage`` and ``strides``.
    features_per_stage : Tuple[int, ...]
        Number of channels at each encoder stage.  The last element
        corresponds to the bottleneck.
    conv_op : Type[nn.Module]
        Convolution operator class (usually ``nn.Conv3d``).
    kernel_sizes, strides, n_blocks_per_stage, n_conv_per_stage_decoder :
        Per‑stage settings copied from the nnUNet planner.  See the
        ``ResidualEncoderUNet`` constructor for details.
    num_classes : int
        Number of segmentation output channels (including background).
    cls_num_classes : int
        Number of classification categories (subtypes).  Defaults to 3.
    deep_supervision : bool, default True
        Whether to return a list of segmentation outputs for deep supervision.
    task_mode : str, default 'both'
        Determines which head(s) participate in gradient computation:
            'seg_only' : only segmentation head updated;
            'cls_only' : only classification head updated;
            'both'     : both heads updated simultaneously.
    """

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Tuple[int, ...],
        conv_op: Type[nn.Module],
        kernel_sizes: Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
        strides: Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
        n_blocks_per_stage: Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
        cls_num_classes: int = 3,
        conv_bias: bool = True,
        norm_op: Type[nn.Module] = nn.InstanceNorm3d,
        norm_op_kwargs: dict | None = None,
        dropout_op: Type[nn.Module] | None = None,
        dropout_op_kwargs: dict | None = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict | None = None,
        deep_supervision: bool = True,
        block: Type[nn.Module] = None,
        stem_channels: int | None = None,
        cls_query_num: int = 8,
        cls_dropout: float = 0.1,
        use_cross_attention: bool = False,
        cls_num_heads: int = 4,
        task_mode: str = 'both',
    ):
        super().__init__()

        # Late import to avoid circular dependency if this file is imported before nnUNet
        from dynamic_network_architectures.building_blocks.residual import (
            StackedResidualBlocks,
            BasicBlockD,
        )
        from dynamic_network_architectures.building_blocks.simple_conv_blocks import (
            StackedConvBlocks,
        )
        from dynamic_network_architectures.building_blocks.helper import (
            get_matching_convtransp,
        )

        if block is None:
            block = BasicBlockD

        assert task_mode in ('both', 'seg_only', 'cls_only')
        self.task_mode = task_mode
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        self.n_stages = n_stages
        self.deep_supervision = False
        self.decoder = DummyDecoder()

        # Handle scalar inputs for per‑stage parameters
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'inplace': True}

        # ------------------------------------------------------------------
        # Encoder (ResNet‑M backbone)
        # ------------------------------------------------------------------
        self.conv_encoder_blocks = nn.ModuleList()
        if stem_channels is None:
            stem_channels = features_per_stage[0]

        # Stem block
        self.conv_encoder_blocks.append(
            StackedResidualBlocks(
                n_blocks_per_stage[0],
                conv_op,
                input_channels,
                stem_channels,
                kernel_sizes[0],
                strides[0],
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                block=block,
            )
        )
        # Subsequent stages
        for s in range(1, n_stages):
            self.conv_encoder_blocks.append(
                StackedResidualBlocks(
                    n_blocks_per_stage[s],
                    conv_op,
                    features_per_stage[s - 1],
                    features_per_stage[s],
                    kernel_sizes[s],
                    strides[s],
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    block=block,
                )
            )

        # ------------------------------------------------------------------
        # Decoder for segmentation
        # ------------------------------------------------------------------
        self.transpconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        transpconv_op = get_matching_convtransp(conv_op)
        for s in range(n_stages - 1, 0, -1):
            # upsample from stage s to s-1
            self.transpconvs.append(
                transpconv_op(
                    features_per_stage[s],
                    features_per_stage[s - 1],
                    strides[s],
                    strides[s],
                    bias=conv_bias,
                )
            )
            decoder_input_features = 2 * features_per_stage[s - 1]
            self.decoder_blocks.append(
                StackedConvBlocks(
                    n_conv_per_stage_decoder[s - 1],
                    conv_op,
                    decoder_input_features,
                    features_per_stage[s - 1],
                    kernel_sizes[s - 1],
                    1,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                )
            )
            self.seg_layers.append(
                conv_op(features_per_stage[s - 1], num_classes, 1, 1, 0, bias=True)
            )

        # ------------------------------------------------------------------
        # Classification branch
        # ------------------------------------------------------------------
        bottleneck_channels = features_per_stage[-1]
        total_cls_input_dim = sum(features_per_stage)
        # A lightweight adapter to further process the bottleneck feature map
        self.cls_adapter = nn.Sequential(
            StackedConvBlocks(
                2,
                conv_op,
                bottleneck_channels,
                bottleneck_channels,
                kernel_sizes[-1],
                1,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
            ),
            conv_op(bottleneck_channels, total_cls_input_dim, 1, 1, 0, bias=True),
        )
        # A simple MLP classifier
        self.classification_head = nn.Sequential(
            nn.Linear(total_cls_input_dim, total_cls_input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(cls_dropout),
            nn.Linear(total_cls_input_dim // 2, cls_num_classes),
        )

    # -- segmentation forward pass with deep supervision --
    def _forward_seg_branch(self, bottleneck_features: torch.Tensor, skips: List[torch.Tensor]):
        seg_outputs = []
        x_dec = bottleneck_features
        for i in range(len(self.decoder_blocks)):
            skip_connection = skips[-(i + 2)]
            x_dec = self.transpconvs[i](x_dec)
            x_dec = torch.cat((x_dec, skip_connection), dim=1)
            x_dec = self.decoder_blocks[i](x_dec)
            seg_outputs.append(self.seg_layers[i](x_dec))
        if self.deep_supervision:
            return seg_outputs[::-1]
        return seg_outputs[-1]

    # -- classification forward pass --
    def _forward_cls_branch(self, bottleneck_features: torch.Tensor):
        adapted = self.cls_adapter(bottleneck_features)
        cls_vec = adapted.mean(dim=(2, 3, 4))  # global average pooling
        return self.classification_head(cls_vec)

    # -- main forward --
    def forward(self, x: torch.Tensor, task_mode: str | None = None):
        mode = task_mode if task_mode is not None else self.task_mode
        assert mode in ('both', 'seg_only', 'cls_only')
        skips: List[torch.Tensor] = []
        x_enc = x
        for i in range(self.n_stages):
            x_enc = self.conv_encoder_blocks[i](x_enc)
            skips.append(x_enc)
        bottleneck_features = skips[-1]

        # segmentation branch
        if mode == 'cls_only':
            # disable gradients for segmentation when training classification only
            with torch.no_grad():
                seg_output = self._forward_seg_branch(bottleneck_features, skips)
        else:
            seg_output = self._forward_seg_branch(bottleneck_features, skips)

        # classification branch
        if mode == 'seg_only':
            # forward classification head without gradients
            with torch.no_grad():
                cls_output = self._forward_cls_branch(bottleneck_features)
        elif mode == 'cls_only':
            # detach encoder features to prevent encoder updates
            cls_output = self._forward_cls_branch(bottleneck_features.detach())
        else:
            cls_output = self._forward_cls_branch(bottleneck_features)
        return seg_output, cls_output


# -----------------------------------------------------------------------------
# 2. Custom experiment planner
#
# We subclass the official residual encoder planner and simply override the
# network class and architecture keyword arguments to include our multi‑task
# network.  Most of the planning logic (patch size, batch size, pooling
# topology, etc.) remains unchanged.
# -----------------------------------------------------------------------------

from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import (
    nnUNetPlannerResEncM,
)


class nnUNetPlannerResEncM_MTL(nnUNetPlannerResEncM):
    """
    Plan for the multi‑task ResNet‑M network.

    This planner behaves identically to the standard ``nnUNetPlannerResEncM``
    except that it instructs nnUNet to instantiate our multi‑task network
    instead of the default ResidualEncoderUNet.  It also injects the
    ``cls_num_classes`` and ``task_mode`` arguments into the architecture
    dictionary so they are available when the network is built.
    """

    def get_plans_for_configuration(
        self,
        spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
        median_shape: Union[np.ndarray, Tuple[int, ...]],
        data_identifier: str,
        approximate_n_voxels_dataset: float,
        _cache: dict,
    ) -> dict:
        # First call the parent to produce a baseline plan
        plan = super().get_plans_for_configuration(
            spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache
        )
        # Now override the network class and add classification specifics
        arch = plan['architecture']
        # Replace the class name with our implementation
        arch['network_class_name'] = (
            f"{__name__}.ResNet_MTL_nnUNet"
        )
        # Insert classification parameters.  We set cls_num_classes to 3
        # (subtypes 0/1/2) and default task_mode to 'both' so both heads are
        # trained concurrently.  Additional kwargs can be specified by editing
        # this dictionary.
        arch['arch_kwargs']['cls_num_classes'] = 3
        arch['arch_kwargs']['task_mode'] = 'both'
        # Ensure PyTorch can resolve our network class at runtime.  These
        # imported names require dynamic import via pydoc.locate when the
        # trainer constructs the network.
        if 'nonlin' not in arch['_kw_requires_import']:
            arch['_kw_requires_import'] = (*arch['_kw_requires_import'], 'nonlin')
        return plan
# logger
import nnunetv2.training.logging.nnunet_logger as logger_base
class nnUNetLoggerMT(logger_base.nnUNetLogger):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.my_fantastic_logging.setdefault('val_loss_seg', [])
        self.my_fantastic_logging.setdefault('val_loss_cls', [])
        self.my_fantastic_logging.setdefault('val_acc_cls', [])
        self.my_fantastic_logging.setdefault('val_accuracy', [])
        self.my_fantastic_logging.setdefault('val_macro_f1', [])
        self.my_fantastic_logging.setdefault('loss_seg', [])
        self.my_fantastic_logging.setdefault('loss_cls', [])

# -----------------------------------------------------------------------------
# 3. Custom trainer
#
# We extend the standard nnUNetTrainer to support multi‑task learning.  The
# modifications are deliberately light: we reuse the vast majority of the
# original training loop, data loading and augmentation.  The key changes are:
#
#  * When instantiating the network we always use our multi‑task model.
#  * The trainer reads the classification labels from the case identifiers and
#    constructs a weighted cross entropy loss to address class imbalance.
#  * ``train_step`` computes both segmentation and classification losses and
#    combines them with epoch‑dependent weights (early epochs emphasise
#    classification).
#  * ``validation_step`` returns both segmentation and classification
#    predictions; ``on_validation_epoch_end`` computes classification metrics.
#
# If you wish to adjust loss weights, training schedule or deep supervision
# behaviour, edit the constants ``INITIAL_LAMBDA_SEG``, ``INITIAL_LAMBDA_CLS``
# and the epoch threshold at which segmentation weight decreases.
# -----------------------------------------------------------------------------

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer as _BaseTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.label_handling.label_handling import (
    determine_num_input_channels,
)
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from sklearn.metrics import f1_score, accuracy_score


class nnUNetTrainerResEncMMTL(_BaseTrainer):
    """
    Multi‑task trainer for ResNet‑M based nnUNet models.

    This class injects classification behaviour into the standard nnUNet training
    pipeline.  It supports three modes of operation via ``task_mode``:
      * 'seg_only' : train segmentation head only
      * 'cls_only' : train classification head only
      * 'both'     : train both heads concurrently

    By default the trainer runs in 'both' mode.  Classification labels are
    inferred from the second underscore‑separated element of each case
    identifier (e.g. ``quiz_2_057`` → subtype 2).  Class weights are computed
    from the distribution of subtypes in the training fold and applied to the
    classification cross entropy loss.
    """

    # Default weighting scheme: emphasise classification early
    INITIAL_LAMBDA_SEG = 0.3
    INITIAL_LAMBDA_CLS = 1.0
    LAMBDA_SEG_AFTER = 0.2
    EPOCH_THRESHOLD = 20

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Override defaults: train both tasks and provide a reasonable patch size for cls_only mode
        self.task_mode = 'both'
        self.cls_patch_size = (96, 160, 224)
        # Placeholders for lambda coefficients; updated each epoch
        self.lambda_seg = self.INITIAL_LAMBDA_SEG
        self.lambda_cls = self.INITIAL_LAMBDA_CLS
        # Classification loss will be initialised in on_train_start
        self.cls_loss = torch.nn.CrossEntropyLoss()
        self.logger = nnUNetLoggerMT()
    
    # We override build_network_architecture to ensure our model is created
    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        task_mode: str,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        # Resolve import strings to classes for the convolution, norm etc.
        for ri in arch_init_kwargs_req_import:
            if arch_init_kwargs[ri] is not None:
                arch_init_kwargs[ri] = pydoc.locate(arch_init_kwargs[ri])
        # Instantiate our network, passing classification parameters from the plan
        # ``cls_num_classes`` and ``task_mode`` come from the plan (set in the planner)
        network = ResNet_MTL_nnUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
            **arch_init_kwargs,
        )
        return network

    def _build_loss(self):
        """
        Use Dice + CrossEntropy for segmentation loss.  This mirrors the base
        trainer.  Deep supervision weights are applied as usual.
        """
        loss = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.plans_manager.get_label_manager(self.dataset_json).ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )
        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)
        if self.enable_deep_supervision:
            deep_supervision_scales = list(
                list(i) for i in 1 / np.cumprod(np.vstack(self.configuration_manager.pool_op_kernel_sizes), axis=0)
            )[:-1]
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def on_train_start(self):
        # Run parent's initialisation (build network, dataloaders etc.)
        super().on_train_start()
        # Compute class weights from training keys.  Identifiers are strings
        # like 'quiz_0_041'; the subtype (0,1,2) is in position 1 after splitting
        tr_labels = None
        try:
            tr_labels = [int(k.split('_')[1]) for k in self.tr_keys]
        except Exception as e:
            self.print_to_log_file(
                f"WARNING: Could not parse classification labels from training identifiers."
                f" Unweighted classification loss will be used. Error: {e}"
            )
        if tr_labels is not None:
            # Determine number of classification categories from the model
            net = self.network.module if self.is_ddp else self.network
            try:
                num_classes = net.classification_head[-1].out_features
            except Exception as e:
                self.print_to_log_file(
                    f"WARNING: Could not determine cls_num_classes from network: {e}."
                    " Unweighted classification loss will be used."
                )
                num_classes = None
            if num_classes is not None:
                counts = np.bincount(tr_labels, minlength=num_classes)
                if np.any(counts == 0):
                    self.print_to_log_file(
                        f"WARNING: Some classes missing in training fold: {np.where(counts==0)[0]}."
                        " Unweighted classification loss will be used."
                    )
                else:
                    n_samples = len(tr_labels)
                    weights = n_samples / (num_classes * counts)
                    class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
                    self.cls_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
                    self.print_to_log_file(
                        f"Applied weighted CrossEntropyLoss to classification head."
                    )
        # Ensure deep supervision is enabled on the network
        self.set_deep_supervision_enabled(self.enable_deep_supervision)

    def get_dataloaders(self):
        """
        Use the base implementation but always draw ``class_label`` from the
        case identifier.  We override this method to extend the batches with
        the subtype index.  nnUNetDataLoader yields a dict with keys
        ``data`` and ``target``; we add ``class_label`` here.
        """
        # Grab the existing dataloaders from the base class
        mt_gen_train, mt_gen_val = super().get_dataloaders()
        return mt_gen_train, mt_gen_val

        # Wrap the generators to attach classification labels on the fly
    def add_cls_label(self, batch):
        """
        从 batch['keys'] 解析分类标签，存到 batch['cls_label'] 里。
        keys 形如 'quiz_<class>_<id>'。
        """
        keys = batch['keys']          # list of length B, 每个是 case_identifier
        cls_labels = []
        for k in keys:
            # 确保只取前面名字，不带后缀
            # 典型情况: k = 'quiz_2_045' 或 'quiz_2_045_0000'
            name = os.path.basename(k)
            name = name.replace('.nii.gz', '')
            parts = name.split('_')
            # ['quiz', '<class>', '<id>', ...]
            cls = int(parts[1])
            cls_labels.append(cls)

        # 这里先用 numpy 存，后面 run_iteration 统一转 torch
        batch['cls_label'] = np.asarray(cls_labels, dtype=np.int64)
        return batch


    def train_step(self, batch: dict) -> dict:
        batch = self.add_cls_label(batch)
        data = batch['data'].to(self.device, non_blocking=True)
        target_seg = batch['target']
        target_cls = torch.from_numpy(batch['cls_label']).long().to(self.device, non_blocking=True)
        if isinstance(target_seg, list):
            target_seg = [i.to(self.device, non_blocking=True) for i in target_seg]
        else:
            target_seg = target_seg.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # Set loss weights depending on epoch
        if self.current_epoch < self.EPOCH_THRESHOLD:
            self.lambda_seg = self.INITIAL_LAMBDA_SEG
            self.lambda_cls = self.INITIAL_LAMBDA_CLS
        else:
            self.lambda_seg = self.LAMBDA_SEG_AFTER
            self.lambda_cls = self.INITIAL_LAMBDA_CLS

        # Forward pass.  Mixed precision is handled by the base class
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output_seg, output_cls = self.network(data)
            # Segmentation loss (handles deep supervision internally)
            l_seg = self.loss(output_seg, target_seg)
            # Classification loss
            l_cls = self.cls_loss(output_cls, target_cls)
            # Combine losses according to current weights
            loss = self.lambda_seg * l_seg + self.lambda_cls * l_cls

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {
            'loss': loss.detach().cpu().numpy(),
            'loss_seg': l_seg.detach().cpu().numpy(),
            'loss_cls': l_cls.detach().cpu().numpy(),
        }
    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(torch.dist.get_world_size())]
            torch.dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])

        self.logger.log('train_losses', loss_here, self.current_epoch)
        self.logger.log('loss_seg', np.mean(outputs['loss_seg']), self.current_epoch)
        self.logger.log('loss_cls', np.mean(outputs['loss_cls']), self.current_epoch)
        
    def validation_step(self, batch: dict) -> dict:
        batch = self.add_cls_label(batch)
        data = batch['data'].to(self.device, non_blocking=True)
        target_seg = batch['target']
        target_cls = torch.from_numpy(batch['cls_label']).long().to(self.device, non_blocking=True)
        if isinstance(target_seg, list):
            target_seg = [i.to(self.device, non_blocking=True) for i in target_seg]
        else:
            target_seg = target_seg.to(self.device, non_blocking=True)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output_seg, output_cls = self.network(data)
            l_seg = self.loss(output_seg, target_seg) if self.task_mode != 'cls_only' else torch.zeros(
                1, device=self.device, dtype=torch.float32
            )
            l_cls = self.cls_loss(output_cls, target_cls) if self.task_mode != 'seg_only' else torch.zeros(
                1, device=self.device, dtype=torch.float32
            )
            # Combine for logging; actual optimisation happens in train_step
            if self.task_mode == 'seg_only':
                loss = l_seg
            elif self.task_mode == 'cls_only':
                loss = l_cls
            else:
                # same weights as in train_step
                if self.current_epoch < self.EPOCH_THRESHOLD:
                    lambda_seg = self.INITIAL_LAMBDA_SEG
                    lambda_cls = self.INITIAL_LAMBDA_CLS
                else:
                    lambda_seg = self.LAMBDA_SEG_AFTER
                    lambda_cls = self.INITIAL_LAMBDA_CLS
                loss = lambda_seg * l_seg + lambda_cls * l_cls

        # Deep supervision: only highest resolution used for metrics
        if self.enable_deep_supervision:
            output_seg_highres = output_seg[0]
            target_seg_highres = target_seg[0]
        else:
            output_seg_highres = output_seg
            target_seg_highres = target_seg
        # One‑hot predictions for dice calculation
        axes = [0] + list(range(2, output_seg_highres.ndim))
        if self.plans_manager.get_label_manager(self.dataset_json).has_regions:
            # Region‑based; use sigmoid + threshold
            pred_seg_onehot = (torch.sigmoid(output_seg_highres) > 0.5).long()
        else:
            out_argmax = output_seg_highres.argmax(1)[:, None]
            pred_seg_onehot = torch.zeros(
                output_seg_highres.shape,
                device=output_seg_highres.device,
                dtype=torch.float32,
            )
            pred_seg_onehot.scatter_(1, out_argmax, 1)
        # Handle ignore label if present
        label_manager = self.plans_manager.get_label_manager(self.dataset_json)
        if label_manager.has_ignore_label:
            if not label_manager.has_regions:
                mask = (target_seg_highres != label_manager.ignore_label).float()
                target_seg_highres = target_seg_highres.clone()
                target_seg_highres[target_seg_highres == label_manager.ignore_label] = 0
            else:
                if target_seg_highres.dtype == torch.bool:
                    mask = ~target_seg_highres[:, -1:]
                else:
                    mask = 1 - target_seg_highres[:, -1:]
                target_seg_highres = target_seg_highres[:, :-1]
        else:
            mask = None
        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
        tp, fp, fn, _ = get_tp_fp_fn_tn(
            pred_seg_onehot,
            target_seg_highres,
            axes=axes,
            mask=mask,
        )
        tp = tp.detach().cpu().numpy()
        fp = fp.detach().cpu().numpy()
        fn = fn.detach().cpu().numpy()
        if not label_manager.has_regions:
            tp = tp[1:]
            fp = fp[1:]
            fn = fn[1:]
        return {
            'loss': loss.detach().cpu().numpy(),
            'loss_seg': l_seg.detach().cpu().numpy(),
            'loss_cls': l_cls.detach().cpu().numpy(),
            'tp_hard': tp,
            'fp_hard': fp,
            'fn_hard': fn,
            'cls_pred': output_cls.detach().cpu().numpy(),
            'cls_target': target_cls.detach().cpu().numpy(),
        }

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        # Aggregate outputs across processes if DDP
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)
        if self.is_ddp:
            world_size = torch.distributed.get_world_size()
            # reduce segmentation counts
            all_tp = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(all_tp, tp)
            tp = np.vstack([i[None] for i in all_tp]).sum(0)
            all_fp = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(all_fp, fp)
            fp = np.vstack([i[None] for i in all_fp]).sum(0)
            all_fn = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(all_fn, fn)
            fn = np.vstack([i[None] for i in all_fn]).sum(0)
            # reduce losses
            losses_val = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
            losses_seg_val = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(losses_seg_val, outputs_collated['loss_seg'])
            loss_seg_here = np.vstack(losses_seg_val).mean()
            losses_cls_val = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(losses_cls_val, outputs_collated['loss_cls'])
            loss_cls_here = np.vstack(losses_cls_val).mean()
            # reduce classification predictions/targets
            cls_preds_all = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(cls_preds_all, outputs_collated['cls_pred'])
            cls_preds = np.concatenate([item for sublist in cls_preds_all for item in sublist])
            cls_targets_all = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(cls_targets_all, outputs_collated['cls_target'])
            cls_targets = np.concatenate([item for sublist in cls_targets_all for item in sublist])
        else:
            loss_here = np.mean(outputs_collated['loss'])
            loss_seg_here = np.mean(outputs_collated['loss_seg'])
            loss_cls_here = np.mean(outputs_collated['loss_cls'])
            cls_preds = np.concatenate(outputs_collated['cls_pred'])
            cls_targets = np.concatenate(outputs_collated['cls_target'])
        # Compute segmentation Dice
        dice_per_class = [2 * i / (2 * i + j + k) if (2 * i + j + k) > 0 else 0 for i, j, k in zip(tp, fp, fn)]
        mean_fg_dice = np.nanmean(dice_per_class)
        # Classification metrics
        cls_pred_int = np.argmax(cls_preds, axis=1)
        macro_f1 = f1_score(cls_targets, cls_pred_int, average='macro', zero_division=0)
        accuracy = accuracy_score(cls_targets, cls_pred_int)
        # Log all metrics
        self.logger.log('val_losses', loss_here, self.current_epoch)
        self.logger.log('val_loss_seg', loss_seg_here, self.current_epoch)
        self.logger.log('val_loss_cls', loss_cls_here, self.current_epoch)
        self.logger.log('dice_per_class_or_region', dice_per_class, self.current_epoch)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('val_accuracy', accuracy, self.current_epoch)
        self.logger.log('val_macro_f1', macro_f1, self.current_epoch)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)
        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('loss_seg', np.round(self.logger.my_fantastic_logging['loss_seg'][-1], decimals=4))
        self.print_to_log_file('loss_cls', np.round(self.logger.my_fantastic_logging['loss_cls'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

# nnUNetv2_plan_and_preprocess -d 002 -pl nnUNetPlannerResEncM_MTL
# nnUNetv2_train 002 3d_fullres 4 -p nnUNetResEncUNetMPlans_MTL -tr nnUNetTrainerResEncMMTL

# nnUNetv2_train 002 3d_fullres 4 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerResEncMMTL
