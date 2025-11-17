import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Counter, Tuple, Union, List
import torch.nn.functional as F

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from torch import autocast, mode, nn
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.cuda import device_count
from torch import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training import lr_scheduler
from nnunetv2.training.nnUNetTrainer.variants import optimizer
from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.tversky import MemoryEfficientTverskyLoss
from nnunetv2.training.loss.classification_losses import ClassBalancedFocalLoss, LDAMLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

# import my model
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.ResNet_MTL_nnUNet import ResNet_MTL_nnUNet
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
# Add these two if they are missing
from nnunetv2.training.lr_scheduler.warmup import Lin_incr_LRScheduler, PolyLRScheduler_offset

# compute validation score
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False

class nnUNetTrainer(object):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # From https://grugbrain.dev/. Worth a read ya big brains ;-)

        # apex predator of grug is complexity
        # complexity bad
        # say again:
        # complexity very bad
        # you say now:
        # complexity very, very bad
        # given choice between complexity or one on one against t-rex, grug take t-rex: at least grug see t-rex
        # complexity is spirit demon that enter codebase through well-meaning but ultimately very clubbable non grug-brain developers and project managers who not fear complexity spirit demon or even know about sometime
        # one day code base understandable and grug can get work done, everything good!
        # next day impossible: complexity demon spirit has entered code and very dangerous situation!

        # OK OK I am guilty. But I tried.
        # https://www.osnews.com/images/comics/wtfm.jpg
        # https://i.pinimg.com/originals/26/b2/50/26b250a738ea4abc7a5af4d42ad93af0.jpg

        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.local_rank = 0 if not self.is_ddp else dist.get_rank()

        self.device = device

        # print what device we are using
        if self.is_ddp:  # implicitly it's clear that we use cuda in this case
            print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
                  f"{dist.get_world_size()}."
                  f"Setting device to {self.device}")
            self.device = torch.device(type='cuda', index=self.local_rank)
        else:
            if self.device.type == 'cuda':
                # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
                self.device = torch.device(type='cuda', index=0)
            print(f"Using device: {self.device}")

        # --- Global backend & AMP/SDPA preferences ---
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        # SDPA/FlashAttention toggles (PyTorch >= 2.0)
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
        except Exception:
            pass
        # Preferred AMP dtype: BF16 if supported, else FP16
        try:
            self.amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            self.amp_dtype = torch.float16

        # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
        # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
        # need. So let's save the init args
        self.my_init_kwargs = {}
        for k in inspect.signature(self.__init__).parameters.keys():
            self.my_init_kwargs[k] = locals()[k]

        ###  Saving all the init args into class variables for later access
        self.plans_manager = PlansManager(plans)
        self.configuration_manager = self.plans_manager.get_configuration(configuration)
        self.configuration_name = configuration
        self.dataset_json = dataset_json
        self.fold = fold

        ### Setting all the folder names. We need to make sure things don't crash in case we are just running
        # inference and some of the folders may not be defined!
        self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
            if nnUNet_preprocessed is not None else None
        self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
                                       self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
            if nnUNet_results is not None else None
        self.output_folder = join(self.output_folder_base, f'fold_{fold}')

        self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
                                                self.configuration_manager.data_identifier)
        self.dataset_class = None  # -> initialize
        # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
        # be a different configuration in the same plans
        # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
        # "previous_stage" and "next_stage"). Otherwise it won't work!
        self.is_cascaded = self.configuration_manager.previous_stage_name is not None
        self.folder_with_segs_from_previous_stage = \
            join(nnUNet_results, self.plans_manager.dataset_name,
                 self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
                 self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
                if self.is_cascaded else None

        ### Some hyperparameters for you to fiddle with
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.5
        self.probabilistic_oversampling = False
        self.num_iterations_per_epoch = 250 
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 500
        self.current_epoch = 0
        self.enable_deep_supervision = True
        # ('both', 'seg_only', 'cls_only')
        self.task_mode = 'both'
        print(f"Task mode set to {self.task_mode}")
        self.cls_patch_size = (96, 160, 224)

        # ===== BOTH 模式下分类 ROI 课程 & 损失权重（新增默认值） =====
        self.roi_use_gt_epochs     = 40   # 前40个epoch用GT ROI，之后用预测ROI
        self.lesion_label_value    = 2    # target_seg里“病灶”标签的整数值（你的数据是0/1/2→病灶=2）


        # 分类平滑（用于 CrossEntropyLoss）
        self.cls_label_smoothing = 0.05

        # EMA 设置
        self.ema_decay = 0.999
        self.ema_model = None  # 训练时维护一份 EMA 模型，用于验证/保存

        # 任务权重：Softmax 约束（替代 log-variance 加权）
        # 初始化时略偏向分割（分类较弱），避免早期不稳定
        self.task_logits = torch.nn.Parameter(torch.tensor([0.0, 0.0], device=self.device))
        self.task_entropy_reg = 0.0  # 可选：>0 防止塌缩，例如 0.01
        # 任务权重护栏：每个任务的最小权重（Softmax 后再做平滑）
        # 使用加性平滑：w <- (1 - K*floor) * softmax(logits) + floor，确保 w_i >= floor 且和为 1
        self.task_weight_floor = 0.15
        self.use_ldam_switch = True

        self.ldam_switch_epoch_frac = 0.5

        self.cbf_beta = 0.9999
        self.cbf_gamma = 2.0
        self.ldam_max_m = 0.5
        self.ldam_s = 30
        self.cls_counts = None



        ### Dealing with labels/regions
        self.label_manager = self.plans_manager.get_label_manager(dataset_json)
        # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
        # needed for predictions. We do sigmoid in case of (overlapping) regions

        self.num_input_channels = None  # -> self.initialize()
        self.network = None  # -> self.build_network_architecture()
        self.optimizer = self.lr_scheduler = None  # -> self.initialize
        # GradScaler only needed for FP16; BF16 uses wide exponent and does not require scaling
        _use_fp16 = (self.device.type == 'cuda' and self.amp_dtype == torch.float16)
        self.grad_scaler = GradScaler("cuda", enabled=_use_fp16) if self.device.type == 'cuda' else None
        self.loss = None  # -> self.initialize
        ### Simple logging. Don't take that away from me!
        # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
        # logging
        timestamp = datetime.now()
        maybe_mkdir_p(self.output_folder)
        self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                             (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second))
        self.logger = nnUNetLogger()

        ### placeholders
        self.dataloader_train = self.dataloader_val = None  # see on_train_start
        self.tr_keys = self.val_keys = None # for cls imbalence place holder
        ### initializing stuff for remembering things and such
        self._best_ema = None
        # Track best Macro-F1 and save a dedicated checkpoint when improved
        self._best_macro_f1 = None

        ### inference things
        self.inference_allowed_mirroring_axes = None  # this variable is set in
        # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints

        ### checkpoint saving stuff
        self.save_every = 50
        self.disable_checkpointing = False

        self.was_initialized = False

        # Weights & Biases (optional)
        self.use_wandb = True
        self.wandb_run = None
        self.wandb_project = os.environ.get('WANDB_PROJECT', 'nnunetv2-quiz-3dct')
        self.wandb_entity = os.environ.get('WANDB_ENTITY', None)

        self.print_to_log_file("\n#######################################################################\n"
                               "Please cite the following paper when using nnU-Net:\n"
                               "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
                               "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
                               "Nature methods, 18(2), 203-211.\n"
                               "#######################################################################\n",
                               also_print_to_console=True, add_timestamp=False)

    def initialize(self):
        if not self.was_initialized:
            ## DDP batch size and oversampling can differ between workers and needs adaptation
            # we need to change the batch size in DDP because we don't use any of those distributed samplers
            self._set_batch_size_and_oversample()

            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                task_mode=self.task_mode,
                enable_deep_supervision=self.enable_deep_supervision,

            ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)
            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.seg_loss = self._build_loss()
            # new classification loss (with label smoothing)
            self.cls_loss = torch.nn.CrossEntropyLoss(label_smoothing=self.cls_label_smoothing)



            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
            # debug
            # names_with_optim = set()
            # for group in self.optimizer.param_groups:
            #     for p in group['params']:
            #         for name, p2 in self.network.named_parameters():
            #             if p2 is p:
            #                 names_with_optim.add(name)
            #                 break

            # print('classification_head in optimizer?:', any('classification_head' in n for n in names_with_optim))
            # end debug
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def _get_model_for_ema(self):
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        return mod

    def _init_ema(self):
        if self.ema_model is None:
            src = self._get_model_for_ema()
            self.ema_model = deepcopy(src).to(self.device)
            for p in self.ema_model.parameters():
                p.requires_grad = False
            self.ema_model.eval()

    def _update_ema(self):
        if self.ema_model is None:
            self._init_ema()
        d = self.ema_decay
        src = self._get_model_for_ema()
        with torch.no_grad():
            for ema_p, p in zip(self.ema_model.parameters(), src.parameters()):
                ema_p.data.mul_(d).add_(p.data, alpha=(1. - d))
            # 同步 buffers（如 BN 统计量）
            for ema_b, b in zip(self.ema_model.buffers(), src.buffers()):
                ema_b.mul_(d).add_(b, alpha=(1. - d))

    def _do_i_compile(self):
        # new default: compile is enabled!

        # compile does not work on mps
        if self.device == torch.device('mps'):
            if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
                self.print_to_log_file("INFO: torch.compile disabled because of unsupported mps device")
            return False

        # CPU compile crashes for 2D models. Not sure if we even want to support CPU compile!? Better disable
        if self.device == torch.device('cpu'):
            if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
                self.print_to_log_file("INFO: torch.compile disabled because device is CPU")
            return False

        # default torch.compile doesn't work on windows because there are apparently no triton wheels for it
        # https://discuss.pytorch.org/t/windows-support-timeline-for-torch-compile/182268/2
        if os.name == 'nt':
            if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
                self.print_to_log_file("INFO: torch.compile disabled because Windows is not natively supported. If "
                                       "you know what you are doing, check https://discuss.pytorch.org/t/windows-support-timeline-for-torch-compile/182268/2")
            return False

        if 'nnUNet_compile' not in os.environ.keys():
            return True
        else:
            return os.environ['nnUNet_compile'].lower() in ('true', '1', 't')

    def _save_debug_information(self):
        # saving some debug information
        if self.local_rank == 0:
            dct = {}
            for k in self.__dir__():
                if not k.startswith("__"):
                    if not callable(getattr(self, k)) or k in ['loss', ]:
                        dct[k] = str(getattr(self, k))
                    elif k in ['network', ]:
                        dct[k] = str(getattr(self, k).__class__.__name__)
                    else:
                        # print(k)
                        pass
                if k in ['dataloader_train', 'dataloader_val']:
                    if hasattr(getattr(self, k), 'generator'):
                        dct[k + '.generator'] = str(getattr(self, k).generator)
                    if hasattr(getattr(self, k), 'num_processes'):
                        dct[k + '.num_processes'] = str(getattr(self, k).num_processes)
                    if hasattr(getattr(self, k), 'transform'):
                        dct[k + '.transform'] = str(getattr(self, k).transform)
            import subprocess
            hostname = subprocess.getoutput(['hostname'])
            dct['hostname'] = hostname
            torch_version = torch.__version__
            if self.device.type == 'cuda':
                gpu_name = torch.cuda.get_device_name()
                dct['gpu_name'] = gpu_name
                cudnn_version = torch.backends.cudnn.version()
            else:
                cudnn_version = 'None'
            dct['device'] = str(self.device)
            dct['torch_version'] = torch_version
            dct['cudnn_version'] = cudnn_version
            save_json(dct, join(self.output_folder, "debug.json"))

    # @staticmethod
    # def build_network_architecture(architecture_class_name: str,
    #                                  arch_init_kwargs: dict,
    #                                  arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
    #                                  num_input_channels: int,
    #                                  num_output_channels: int,
    #                                  task_mode: str,
    #                                  enable_deep_supervision: bool = True
    #                                  ) -> nn.Module:
    #     """
    #     此方法被重写，以直接加载自定义的 ResNet_MTL_nnUNet 模型，
    #     模仿 nnXNetTrainer 的方式。
    #     """
    #     import pydoc  # 像范例中一样，在这里导入 pydoc

    #     # 1. 复制从 plans 文件加载的架构参数
    #     architecture_kwargs = dict(**arch_init_kwargs)

    #     # 2. 将字符串 (例如 'nn.InstanceNorm3d') 转换为类
    #     for ri in arch_init_kwargs_req_import:
    #         if architecture_kwargs[ri] is not None:
    #             architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    #     # 3. 直接实例化你的自定义模型
    #     # 注意：
    #     # - `num_output_channels` 被传递给 `num_classes` (分割头)
    #     # - `cls_num_classes` 预计存在于 `architecture_kwargs` 中 (从 plans.json 加载)
    #     network = ResNet_MTL_nnUNet(
    #         input_channels=num_input_channels,
    #         num_classes=num_output_channels,  # 分割头的类别数
    #         deep_supervision=enable_deep_supervision,
    #         cls_num_classes=3,
    #         task_mode=task_mode,
    #         **architecture_kwargs  # 解包所有来自 plans 文件的参数
    #                                  # (例如 n_stages, features_per_stage, block, cls_num_classes 等)
    #     )

    #     return network

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                arch_init_kwargs: dict,
                                arch_init_kwargs_req_import,
                                num_input_channels: int,
                                num_output_channels: int,
                                task_mode: str,
                                enable_deep_supervision: bool = True) -> nn.Module:
        import pydoc

        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

        # arch_init_kwargs 里其实就是 ResidualEncoderUNet 那些字段：
        # n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_blocks_per_stage, ...
        net = ResNet_MTL_nnUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,         # seg 头类别数
            deep_supervision=enable_deep_supervision,
            task_mode=task_mode,
            cls_num_classes=3, 
            lesion_channel_idx = 2,                                            # 你自己的分类数
            **architecture_kwargs
        )

        return net

    def _compute_voxel_cls_loss(self, logits: torch.Tensor, target_cls_map: torch.Tensor) -> torch.Tensor:
        """
        logits:
            - 新设计: [B, C, D, H, W]，C = cls_num_classes (3)
        target_cls_map:
            - 新设计: [B, 1, D, H, W] 或 [B, D, H, W]
              约定: 0 = 背景(胰腺+其他), 1..C = lesion subtype 0/1/2
            - 兼容旧版: [B]，case-level label (0/1/2)，会 fallback 到全局 pooling 做 CE

        训练逻辑（新设计）:
            只在 target_cls_map > 0 的 voxel 上做 CE，
            并且把标签减 1，使得 subtype 0/1/2 对应 0/1/2。
        """

        # ---------------------------
        # 兼容旧版: target 还是 [B] 的情况
        # ---------------------------
        if target_cls_map.ndim == 1:
            # logits 可能是 [B, C, D, H, W]，我们做一个全局平均得到 [B, C]
            if logits.ndim == 5:
                logits_global = logits.mean(dim=(2, 3, 4))  # [B, C]
            elif logits.ndim == 2:
                logits_global = logits
            else:
                raise RuntimeError(f"Unexpected logits shape {logits.shape} for old-style class targets [B].")

            return self.cls_loss(logits_global, target_cls_map.long())

        # ---------------------------
        # 新设计: voxel-wise subtype map
        # ---------------------------
        if logits.ndim != 5:
            raise RuntimeError(f"Expected cls logits [B, C, D, H, W], got {logits.shape}")

        # target [B, 1, D, H, W] -> [B, D, H, W]
        if target_cls_map.ndim == 5 and target_cls_map.shape[1] == 1:
            target_map = target_cls_map[:, 0]
        elif target_cls_map.ndim == 4:
            target_map = target_cls_map
        else:
            raise RuntimeError(f"Unexpected target_cls_map shape {target_cls_map.shape}")

        B, C, D, H, W = logits.shape

        # 展平成 [N_voxel, C] / [N_voxel]
        logits_flat = logits.permute(0, 2, 3, 4, 1).reshape(-1, C)  # [N, C]
        targets_flat = target_map.reshape(-1).long()                # [N]

        # 只在 lesion voxel 上训练: target > 0
        lesion_mask = targets_flat > 0
        if not lesion_mask.any():
            # 这个 batch 没有任何 lesion voxel，返回 0 loss（不影响梯度）
            return torch.zeros((), device=logits.device, dtype=logits.dtype)

        logits_valid = logits_flat[lesion_mask]        # [M, C]
        # 1..C  ->  0..C-1
        targets_valid = targets_flat[lesion_mask] - 1  # [M]

        return self.cls_loss(logits_valid, targets_valid)


    def _voxel_level_to_case_label(self, output_cls: torch.Tensor, target_cls_map: torch.Tensor):
        """
        根据体素级 subtype map，按 voxel count 决策 case-level label（pred & gt 都这么算）

        output_cls: [B, C_cls, D, H, W] (logits)
        target_cls_map: [B, D, H, W] 或 [B,1,D,H,W]，lesion 体素 0..C_cls-1，背景/胰腺 -1
        返回：
            case_pred_labels: List[int], len = 有效病例数
            case_true_labels: List[int], len 同上
        """
        if target_cls_map is None:
            return [], []

        if target_cls_map.ndim == 5:
            target_cls_map = target_cls_map[:, 0]

        # voxel-level 预测
        pred_voxel = output_cls.argmax(1)  # [B, D,H,W]

        B = pred_voxel.shape[0]
        num_cls = self.network.cls_num_classes if hasattr(self.network, 'cls_num_classes') else int(output_cls.shape[1])

        case_pred_labels = []
        case_true_labels = []

        for b in range(B):
            gt_map_b = target_cls_map[b]          # [D,H,W]
            valid_mask_b = gt_map_b >= 0          # lesion 区域

            if not valid_mask_b.any():
                continue  # 这个 case 没有 lesion，直接跳过（不计入分类指标）

            gt_vals = gt_map_b[valid_mask_b].view(-1).long()          # [N]
            pred_vals = pred_voxel[b][valid_mask_b].view(-1).long()   # [N]

            # gt & pred 分别按 voxel count 投票
            gt_counts = torch.bincount(gt_vals, minlength=num_cls)
            pred_counts = torch.bincount(pred_vals, minlength=num_cls)

            gt_label = int(gt_counts.argmax().item())
            pred_label = int(pred_counts.argmax().item())

            case_true_labels.append(gt_label)
            case_pred_labels.append(pred_label)

        return case_pred_labels, case_true_labels

    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
                self.configuration_manager.pool_op_kernel_sizes), axis=0))[:-1]
        else:
            deep_supervision_scales = None  # for train and val_transforms
        return deep_supervision_scales

    def _set_batch_size_and_oversample(self):
        if not self.is_ddp:
            # set batch size to what the plan says, leave oversample untouched
            self.batch_size = self.configuration_manager.batch_size
        else:
            # batch size is distributed over DDP workers and we need to change oversample_percent for each worker

            world_size = dist.get_world_size()
            my_rank = dist.get_rank()

            global_batch_size = self.configuration_manager.batch_size
            assert global_batch_size >= world_size, 'Cannot run DDP if the batch size is smaller than the number of ' \
                                                    'GPUs... Duh.'

            batch_size_per_GPU = [global_batch_size // world_size] * world_size
            batch_size_per_GPU = [batch_size_per_GPU[i] + 1
                                  if (batch_size_per_GPU[i] * world_size + i) < global_batch_size
                                  else batch_size_per_GPU[i]
                                  for i in range(len(batch_size_per_GPU))]
            assert sum(batch_size_per_GPU) == global_batch_size

            sample_id_low = 0 if my_rank == 0 else np.sum(batch_size_per_GPU[:my_rank])
            sample_id_high = np.sum(batch_size_per_GPU[:my_rank + 1])

            # This is how oversampling is determined in DataLoader
            # round(self.batch_size * (1 - self.oversample_foreground_percent))
            # We need to use the same scheme here because an oversample of 0.33 with a batch size of 2 will be rounded
            # to an oversample of 0.5 (1 sample random, one oversampled). This may get lost if we just numerically
            # compute oversample
            oversample = [True if not i < round(global_batch_size * (1 - self.oversample_foreground_percent)) else False
                          for i in range(global_batch_size)]

            if sample_id_high / global_batch_size < (1 - self.oversample_foreground_percent):
                oversample_percent = 0.0
            elif sample_id_low / global_batch_size > (1 - self.oversample_foreground_percent):
                oversample_percent = 1.0
            else:
                oversample_percent = sum(oversample[sample_id_low:sample_id_high]) / batch_size_per_GPU[my_rank]

            print("worker", my_rank, "oversample", oversample_percent)
            print("worker", my_rank, "batch_size", batch_size_per_GPU[my_rank])

            self.batch_size = batch_size_per_GPU[my_rank]
            self.oversample_foreground_percent = oversample_percent

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            # If the dataset does not declare an explicit ignore label, we still need to
            # ignore padded voxels (-1) produced by cropping. Use -1 as an implicit ignore
            # for the loss to prevent invalid indexing in one-hot scatter.
            effective_ignore = self.label_manager.ignore_label if self.label_manager.ignore_label is not None else -1
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp,
                                   'alpha': 0.3, 'beta': 0.7, 'focal_gamma': 1.5},
                                  {}, weight_ce=0.3, weight_dice=0.7,
                                  ignore_label=effective_ignore, dice_class=MemoryEfficientTverskyLoss)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self, patch_size=None):
        """
        原版函数 + 支持自定义 patch_size（比如 cls_only 时用 cls_patch_size）
        """
        if patch_size is None:
            patch_size = self.configuration_manager.patch_size

        dim = len(patch_size)
        # todo rotation should be defined dynamically based on patch size (more isotropic patch sizes = more rotation)
        if dim == 2:
            do_dummy_2d_data_aug = False
            # todo revisit this parametrization
            if max(patch_size) / min(patch_size) > 1.5:
                rotation_for_DA = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            else:
                rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            mirror_axes = (0, 1)
        elif dim == 3:
            # order of the axes is determined by spacing, not image size
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
            if do_dummy_2d_data_aug:
                rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            else:
                rotation_for_DA = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError()

        initial_patch_size = get_patch_size(patch_size[-dim:],
                                            rotation_for_DA,
                                            rotation_for_DA,
                                            rotation_for_DA,
                                            (0.85, 1.25))
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]

        self.print_to_log_file(f'do_dummy_2d_data_aug: {do_dummy_2d_data_aug}')
        self.inference_allowed_mirroring_axes = mirror_axes

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes


    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        if self.local_rank == 0:
            timestamp = time()
            dt_object = datetime.fromtimestamp(timestamp)

            if add_timestamp:
                args = (f"{dt_object}:", *args)

            successful = False
            max_attempts = 5
            ctr = 0
            while not successful and ctr < max_attempts:
                try:
                    with open(self.log_file, 'a+') as f:
                        for a in args:
                            f.write(str(a))
                            f.write(" ")
                        f.write("\n")
                    successful = True
                except IOError:
                    print(f"{datetime.fromtimestamp(timestamp)}: failed to log: ", sys.exc_info())
                    sleep(0.5)
                    ctr += 1
            if also_print_to_console:
                print(*args)
        elif also_print_to_console:
            print(*args)

    def print_plans(self):
        if self.local_rank == 0:
            dct = deepcopy(self.plans_manager.plans)
            del dct['configurations']
            self.print_to_log_file(f"\nThis is the configuration used by this "
                                   f"training:\nConfiguration name: {self.configuration_name}\n",
                                   self.configuration_manager, '\n', add_timestamp=False)
            self.print_to_log_file('These are the global plan.json settings:\n', dct, '\n', add_timestamp=False)

    def configure_optimizers(self):
        
        # ⭐️ 关键改动：根据 task_mode 选择 scheduler ⭐️
        if self.task_mode == 'cls_only':
            self.print_to_log_file("Using CLS_ONLY mode: Optimizer=SGD, LR_Scheduler=CosineAnnealingLR")
            # 优化器（可以保持 SGD，或者换成 AdamW）
            optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                        momentum=0.99, nesterov=True)
            # 调度器（使用 Cosine）
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        
        else: # 'seg_only' or 'both'
            enc_params, dec_params, cls_params = [], [], []
            for n, p in self.network.named_parameters():
                if not p.requires_grad:
                    continue
                if 'encoder' in n:
                    enc_params.append(p)
                elif any(k in n for k in ['ct_proj', 'dual_block', 'cls_out']):
                    cls_params.append(p)
                else:
                    dec_params.append(p)
            optimizer = torch.optim.AdamW(
                [
                    {'params': enc_params, 'lr': 3e-4},
                    {'params': dec_params, 'lr': 3e-4},
                    {'params': cls_params, 'lr': 1e-3},   # 分类头更大一点
                    {'params': [self.task_logits], 'lr': 1e-3, 'weight_decay': 0.0},  # 任务权重 logits，不做WD
                ],
                weight_decay=1e-4
    )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        return optimizer, lr_scheduler


    def plot_network_architecture(self):
        if self._do_i_compile():
            self.print_to_log_file("Unable to plot network architecture: nnUNet_compile is enabled!")
            return

        if self.local_rank == 0:
            try:
                # raise NotImplementedError('hiddenlayer no longer works and we do not have a viable alternative :-(')
                # pip install git+https://github.com/saugatkandel/hiddenlayer.git

                # from torchviz import make_dot
                # # not viable.
                # make_dot(tuple(self.network(torch.rand((1, self.num_input_channels,
                #                                         *self.configuration_manager.patch_size),
                #                                        device=self.device)))).render(
                #     join(self.output_folder, "network_architecture.pdf"), format='pdf')
                # self.optimizer.zero_grad()

                # broken.

                import hiddenlayer as hl
                g = hl.build_graph(self.network,
                                   torch.rand((1, self.num_input_channels,
                                               *self.configuration_manager.patch_size),
                                              device=self.device),
                                   transforms=None)
                g.save(join(self.output_folder, "network_architecture.pdf"))
                del g
            except Exception as e:
                self.print_to_log_file("Unable to plot network architecture:")
                self.print_to_log_file(e)

                # self.print_to_log_file("\nprinting the network instead:\n")
                # self.print_to_log_file(self.network)
                # self.print_to_log_file("\n")
            finally:
                empty_cache(self.device)

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = self.dataset_class.get_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            dataset = self.dataset_class(self.preprocessed_dataset_folder,
                                         identifiers=None,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(list(dataset.identifiers)))
                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(dataset.identifiers))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            if any([i in val_keys for i in tr_keys]):
                self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                       'splits.json or ignore if this is intentional.')
        return tr_keys, val_keys

    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys, val_keys = self.do_split()
        self.tr_keys = tr_keys
        self.val_keys = val_keys

        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr = self.dataset_class(self.preprocessed_dataset_folder, tr_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        return dataset_tr, dataset_val

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        # 1) 拿到数据集对象
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        # 2) 先根据 task_mode 准备 patch_size 与 transforms（注意顺序，先定义 transforms 再实例化 dataloader）
        if self.task_mode == 'cls_only':
            patch_size = self.cls_patch_size
            deep_supervision_scales = None

            (rotation_for_DA,
            do_dummy_2d_data_aug,
            _initial_patch_size,   # cls_only 下不用，但占位
            mirror_axes) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size(patch_size)

            tr_transforms = self.get_cls_training_transforms(
                patch_size,
                rotation_for_DA,
                deep_supervision_scales,
                mirror_axes,
                do_dummy_2d_data_aug,
                use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
                is_cascaded=self.is_cascaded,
                foreground_labels=self.label_manager.foreground_labels,
                regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
                ignore_label=self.label_manager.ignore_label
            )
            val_transforms = self.get_validation_transforms(
                deep_supervision_scales,
                is_cascaded=self.is_cascaded,
                foreground_labels=self.label_manager.foreground_labels,
                regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
                ignore_label=self.label_manager.ignore_label
            )

            # 3) 直接用标准 nnUNetDataLoader（无均衡、无 subset_keys）
            dl_tr = nnUNetDataLoader(
                dataset_tr, self.batch_size,
                patch_size,            # initial_patch_size
                patch_size,            # final_patch_size
                self.label_manager,
                task_mode=self.task_mode,
                oversample_foreground_percent=0.0,
                sampling_probabilities=None,
                pad_sides=None,
                probabilistic_oversampling=False,
                transforms=tr_transforms
            )
            dl_val = nnUNetDataLoader(
                dataset_val, self.batch_size,
                patch_size,            # initial_patch_size
                patch_size,            # final_patch_size
                self.label_manager,
                task_mode=self.task_mode,
                oversample_foreground_percent=0.0,
                sampling_probabilities=None,
                pad_sides=None,
                probabilistic_oversampling=False,
                transforms=val_transforms
            )

        else:
            # seg_only / both
            patch_size = self.configuration_manager.patch_size
            deep_supervision_scales = self._get_deep_supervision_scales()

            (rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,     # seg/both 下需要：用于 random crop 区间计算
            mirror_axes) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size(patch_size)

            tr_transforms = self.get_training_transforms(
                patch_size,
                rotation_for_DA,
                deep_supervision_scales,
                mirror_axes,
                do_dummy_2d_data_aug,
                use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
                is_cascaded=self.is_cascaded,
                foreground_labels=self.label_manager.foreground_labels,
                regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
                ignore_label=self.label_manager.ignore_label
            )
            val_transforms = self.get_validation_transforms(
                deep_supervision_scales,
                is_cascaded=self.is_cascaded,
                foreground_labels=self.label_manager.foreground_labels,
                regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
                ignore_label=self.label_manager.ignore_label
            )

            # 3) 直接用标准 nnUNetDataLoader（无均衡、无 subset_keys）
            dl_tr = nnUNetDataLoader(
                dataset_tr, self.batch_size,
                initial_patch_size,            # 👈 训练时随机裁剪的“起始尺寸”
                patch_size,                    # 👈 网络输入的最终尺寸（plans里的 patch_size）
                self.label_manager,
                task_mode=self.task_mode,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
                probabilistic_oversampling=self.probabilistic_oversampling,
                transforms=tr_transforms
            )
            dl_val = nnUNetDataLoader(
                dataset_val, self.batch_size,
                patch_size,                    # 验证通常不做 random crop
                patch_size,
                self.label_manager,
                task_mode=self.task_mode,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
                probabilistic_oversampling=self.probabilistic_oversampling,
                transforms=val_transforms
            )

        # 4) 多线程封装（保持与原先一致）
        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr, transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2), seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.002
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val, transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4), seeds=None,
                pin_memory=self.device.type == 'cuda',
                wait_time=0.002
            )

        # 5) 预热一次，保持与原逻辑一致
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val





    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)
    @staticmethod
    def get_cls_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        """
        为分类任务定制的轻量级数据增强
        移除了 SpatialTransform, GaussianBlur, SimulateLowResolution, Mirroring
        保留了色彩/强度增强
        """
        transforms = []
        
        # --- 伪 2D 逻辑 (如果需要，保持不变) ---
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
            
        # ！！！关键改动：移除或大幅减弱 SpatialTransform ！！！
        # 我们可以保留一个非常温和的旋转，或者干脆去掉
        # 示例：保留小角度旋转和缩放，去掉弹性形变
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, 
                p_elastic_deform=0, # 关闭弹性形变
                p_rotation=0.1,     # 降低旋转概率
                rotation=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi), # 减小旋转角度
                p_scaling=0.1,      # 降低缩放概率
                scaling=(0.9, 1.1), # 减小缩放范围
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False
            )
        )
        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())
        # ----------------------------------------------
        
        # --- 保留强度/噪声增强 ---
        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.05), # 也可以适当减弱噪声
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.85, 1.15)), # 减弱亮度变化
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.85, 1.15)), # 减弱对比度变化
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.8, 1.2)), # 减弱 Gamma
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.15
        ))
        # --- 移除 MirrorTransform ---
        # if mirror_axes is not None and len(mirror_axes) > 0:
        #     transforms.append(
        #         MirrorTransform(
        #             allowed_axes=mirror_axes
        #         )
        #     )
        
        # --- Masking 和 Label 转换 (保持不变) ---
        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        # --- 分类任务不需要这些 ---
        # transforms.append(RemoveLabelTansform(-1, 0))
        # if is_cascaded: ...
        # if regions is not None: ...
        # if deep_supervision_scales is not None: ...

        return ComposeTransforms(transforms)
    @staticmethod
    def get_validation_transforms(
            deep_supervision_scales: Union[List, Tuple, None],
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        if is_cascaded:
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        return ComposeTransforms(transforms)

    def set_deep_supervision_enabled(self, enabled: bool):
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        # 1) 自己加的标志，留着也没问题
        mod.deep_supervision = enabled

        # 2) 真正控制输出的是 decoder 的 deep_supervision
        if hasattr(mod, 'decoder'):
            mod.decoder.deep_supervision = enabled


    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()

        self.dataloader_train, self.dataloader_val = self.get_dataloaders()

        self.print_to_log_file("Calculating class weights for imbalanced classification task...")

        if self.tr_keys is None:
            raise RuntimeError("self.tr_keys was not set in get_tr_and_val_datasets. This should not happen.")
        tr_keys = self.tr_keys

        # 1. 从文件名解析全局 subtype（例如 quiz_0_041 → 0）
        try:
            tr_labels = [int(k.split('_')[1]) for k in tr_keys]
        except Exception as e:
            self.print_to_log_file(f"!!! WARNING: Could not parse labels from training keys for weighted loss.")
            self.print_to_log_file(f"!!! Expect identifier like 'quiz_0_041'. Using UNWEIGHTED cls loss. Error: {e}")
            tr_labels = None

        if tr_labels is not None:
            # 2. 从网络拿到分类头类别数（现在是 voxel-level conv head）
            if self.is_ddp:
                net = self.network.module
            else:
                net = self.network

            try:
                head = getattr(net, 'classification_head', None)
                if head is not None:
                    if isinstance(head, nn.Sequential):
                        last = list(head.children())[-1]
                    else:
                        last = head
                    num_classes = last.out_features
                elif hasattr(net, 'cls_out'):
                    num_classes = net.cls_out.out_features
                elif hasattr(net, 'cls_num_classes'):
                    num_classes = net.cls_num_classes
                else:
                    raise RuntimeError("no head found")
            except Exception as e:
                self.print_to_log_file(f"!!! WARNING: Could not determine num_classes from model: {e}")
                self.print_to_log_file("!!! Using UNWEIGHTED cls loss.")
                num_classes = None


            if num_classes is not None:
                counts = np.bincount(tr_labels, minlength=num_classes)

                if np.any(counts == 0):
                    self.print_to_log_file(f"!!! WARNING: Classes {np.where(counts==0)[0]} have 0 samples in this fold.")
                    self.print_to_log_file("!!! Using UNWEIGHTED cls loss to avoid division by zero.")
                else:
                    # class-balanced weight：n / (K * n_c)
                    n_samples = len(tr_labels)
                    weights = n_samples / (num_classes * counts)

                    class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

                    self.cls_counts = counts.tolist()
                    self.cls_loss = ClassBalancedFocalLoss(self.cls_counts, beta=self.cbf_beta, gamma=self.cbf_gamma)
                    self.print_to_log_file(f"Successfully applied Class-Balanced Focal Loss.")
                    self.print_to_log_file(f"Class counts (this fold): {counts}")

        maybe_mkdir_p(self.output_folder)

        self.set_deep_supervision_enabled(self.enable_deep_supervision)

        self.print_plans()
        empty_cache(self.device)

        if self.local_rank == 0:
            self.dataset_class.unpack_dataset(
                self.preprocessed_dataset_folder,
                overwrite_existing=False,
                num_processes=max(1, round(get_allowed_n_proc_DA() // 2)),
                verify=True)

        if self.is_ddp:
            dist.barrier()

        save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
        save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)
        shutil.copy(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                    join(self.output_folder_base, 'dataset_fingerprint.json'))

        # Ensure classification ROI behavior is consistent train/val
        try:
            net_mod = self.network.module if self.is_ddp else self.network
            setattr(net_mod, 'cls_roi_dilate', True)
            # 同时启用并设置 Top-K ROI 池化默认比例（可被外部覆盖）
            setattr(net_mod, 'use_topk_pool', True)
            setattr(net_mod, 'topk_ratio', 0.2)
        except Exception:
            pass

        # Initialize Weights & Biases (rank 0 only)
        if self.local_rank == 0 and self.use_wandb and _WANDB_AVAILABLE:
            try:
                run_name = f"{self.plans_manager.dataset_name}__{self.__class__.__name__}__{self.configuration_name}__fold{self.fold}"
                self.wandb_run = wandb.init(
                    project=self.wandb_project,
                    entity=self.wandb_entity,
                    name=run_name,
                    config={
                        'dataset': self.plans_manager.dataset_name,
                        'configuration': self.configuration_name,
                        'fold': self.fold,
                        'initial_lr': self.initial_lr,
                        'weight_decay': self.weight_decay,
                        'epochs': self.num_epochs,
                        'task_mode': self.task_mode,
                        'enable_deep_supervision': self.enable_deep_supervision,
                        'oversample_foreground_percent': self.oversample_foreground_percent,
                        'probabilistic_oversampling': self.probabilistic_oversampling,
                        'tversky_alpha': 0.3,
                        'tversky_beta': 0.7,
                        'tversky_gamma': 1.5,
                        'cls_loss': 'CB-Focal',
                        'ldam_switch': getattr(self, 'use_ldam_switch', False),
                    },
                    reinit=False,
                )
                try:
                    wandb.save(self.log_file, policy='now')
                except Exception:
                    pass
            except Exception as e:
                self.print_to_log_file(f"W+B init failed: {e}")

        # produces a pdf in output folder
        self.plot_network_architecture()
        self._save_debug_information()


    def on_train_end(self):
        # dirty hack because on_epoch_end increments the epoch counter and this is executed afterwards.
        # This will lead to the wrong current epoch to be stored
        self.current_epoch -= 1
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        self.current_epoch += 1

        # now we can delete latest
        if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))

        # shut down dataloaders
        old_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            if self.dataloader_train is not None and \
                    isinstance(self.dataloader_train, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_train._finish()
            if self.dataloader_val is not None and \
                    isinstance(self.dataloader_train, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_val._finish()
            sys.stdout = old_stdout

        empty_cache(self.device)
        self.print_to_log_file("Training done.")
        # W&B: upload final log and finish
        if self.local_rank == 0 and self.use_wandb and _WANDB_AVAILABLE and self.wandb_run is not None:
            try:
                art = wandb.Artifact('training_logs', type='log')
                art.add_file(self.log_file)
                wandb.log_artifact(art)
            except Exception:
                pass
            try:
                wandb.finish()
            except Exception:
                pass

    def on_train_epoch_start(self):
        self.network.train()
        # self.lr_scheduler.step(self.current_epoch)
        if isinstance(self.lr_scheduler, (PolyLRScheduler, Lin_incr_LRScheduler, PolyLRScheduler_offset)):
            # These custom schedulers *do* expect the epoch
            self.lr_scheduler.step(self.current_epoch)
        else:
            # Standard PyTorch schedulers (like CosineAnnealingLR) do not
            self.lr_scheduler.step()
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)
        # W&B: log LR per-epoch
        if self.local_rank == 0 and self.use_wandb and _WANDB_AVAILABLE and self.wandb_run is not None:
            try:
                wandb.log({'lr': float(self.optimizer.param_groups[0]['lr'])}, step=self.current_epoch)
            except Exception:
                pass

        if self.use_ldam_switch and (self.cls_counts is not None):
            if not hasattr(self, '_switched_to_ldam'):
                self._switched_to_ldam = False
            if (not self._switched_to_ldam) and (self.current_epoch >= int(self.ldam_switch_epoch_frac * self.num_epochs)):
                beta = self.cbf_beta
                n = np.array(self.cls_counts, dtype=float)
                eff = (1.0 - beta) / (1.0 - np.clip(np.power(beta, n), 1e-12, None))
                drw = (len(n) * eff / eff.sum()).tolist()
                self.cls_loss = LDAMLoss(self.cls_counts, max_m=self.ldam_max_m, s=self.ldam_s, drw_weight=drw)
                self._switched_to_ldam = True
                self.print_to_log_file("Switched classification loss to LDAM+DRW.")


    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target_seg = batch['target']        # seg label (原逻辑不变)
        target_cls_map = batch['class_label']  # 现在可以是 [B] 或 [B,1,D,H,W]

        data = data.to(self.device, non_blocking=True)

        if isinstance(target_seg, list):
            target_seg = [i.to(self.device, non_blocking=True) for i in target_seg]
        else:
            target_seg = target_seg.to(self.device, non_blocking=True)

        target_cls_map = torch.as_tensor(target_cls_map, device=self.device)

        # 统一得到 case-level 的 target（用于 CE），处理 [B] 或 voxel map 的情况
        if target_cls_map.ndim == 1:
            target_cls = target_cls_map.long()
        else:
            # 优先从 keys 解析（batch 中应包含 'keys': ['quiz_0_xxx', ...]）
            keys = batch.get('keys', None)
            target_cls = None
            if keys is not None:
                try:
                    parsed = [int(k.split('_')[1]) for k in keys]
                    target_cls = torch.as_tensor(parsed, device=self.device, dtype=torch.long)
                except Exception:
                    target_cls = None
            if target_cls is None:
                # 回退：从 voxel map 多数投票得到 case label
                tmap = target_cls_map
                if tmap.ndim == 5 and tmap.shape[1] >= 1:
                    tmap = tmap[:, 0]
                elif tmap.ndim != 4:
                    raise RuntimeError(f"Unexpected class_label shape {tuple(tmap.shape)}")
                B = tmap.shape[0]
                flat = tmap.reshape(B, -1).long()
                # 简单取众数（包含 0），若需要忽略 0 可自行调整
                target_cls = torch.mode(flat, dim=1).values.to(torch.long)

        self.optimizer.zero_grad(set_to_none=True)
        mode = self.task_mode

        with autocast(self.device.type, dtype=self.amp_dtype) if self.device.type == 'cuda' else dummy_context():
            # 网络返回 (seg_output, cls_output)
            # task_mode 的具体梯度逻辑在模型内部处理
            roi_mask_for_cls = None

            if mode == 'both':
                # 取最高分辨率的 GT 分割，并规范成 5D (B,1,D,H,W)
                seg_hires = target_seg[0] if isinstance(target_seg, list) else target_seg
                if seg_hires.ndim == 4:
                    seg_hires_5d = seg_hires.unsqueeze(1)        # B×1×DHW
                elif seg_hires.ndim == 5:
                    seg_hires_5d = seg_hires                     # B×C×DHW 或 B×1×DHW
                else:
                    raise RuntimeError(f"Unexpected seg_hires ndim={seg_hires.ndim}, expect 4D or 5D.")

                # 前 N 个 epoch 用 GT ROI，之后用预测 ROI
                roi_use_gt_epochs  = getattr(self, 'roi_use_gt_epochs', 30)

                if self.current_epoch < roi_use_gt_epochs:
                    # 只使用“病灶”作为 ROI：兼容 one-hot 或 labelmap
                    if seg_hires_5d.shape[1] == 1:
                        # labelmap: 取等于病灶标签值的位置为 ROI
                        lesion_val = getattr(self, 'lesion_label_value', 2)
                        roi_mask_for_cls = (seg_hires_5d == lesion_val).float()
                    else:
                        # one-hot: 直接取病灶通道
                        lesion_ch = int(max(0, min(getattr(self.network, 'lesion_channel_idx', 2), seg_hires_5d.shape[1] - 1)))
                        roi_mask_for_cls = (seg_hires_5d[:, lesion_ch:lesion_ch + 1] > 0).float()
                    # 轻微膨胀，给分类更多上下文
                    if getattr(self, 'cls_roi_dilate', True):
                        roi_mask_for_cls = F.max_pool3d(roi_mask_for_cls, kernel_size=3, stride=1, padding=1)
                    roi_mask_for_cls = roi_mask_for_cls.clamp_min(1e-3)
                else:
                    roi_mask_for_cls = None  # 切换到预测 ROI（模型内部会从 seg_output 取前景概率）

            # 传入 roi_mask（B,1,D,H,W 或 None）
            output_seg, output_cls = self.network(data, task_mode=None, roi_mask=roi_mask_for_cls)




            if mode == 'seg_only':
                self.lambda_seg, self.lambda_cls = 1.0, 0.0
                # removed debug stats
                l_seg = self.seg_loss(output_seg, target_seg)
                l_cls = torch.zeros(1, device=self.device, dtype=l_seg.dtype)
                l = l_seg

            elif mode == 'cls_only':
                self.lambda_seg, self.lambda_cls = 0.0, 1.0
                l_seg = torch.zeros(1, device=self.device, dtype=torch.float32)
                logits = output_cls.float()
                # 在 AMP(BF16/FP16) 下直接计算分类损失，BF16 足够稳定
                l_cls = self.cls_loss(logits, target_cls)
                l = l_cls

            else:  # mode == 'both'
                # seg + cls 一起训：Softmax 约束权重
                l_seg = self.seg_loss(output_seg, target_seg)
                logits = output_cls.float()
                l_cls = self.cls_loss(logits, target_cls)
                w = torch.softmax(self.task_logits, dim=0)
                # 加性平滑护栏：保证每个任务权重 >= floor，且总和为 1
                _k = w.shape[0]
                _floor = min(self.task_weight_floor, 1.0 / _k - 1e-6)
                w = w * (1.0 - _k * _floor) + _floor
                l = w[0] * l_seg + w[1] * l_cls
                if self.task_entropy_reg > 0:
                    l = l + self.task_entropy_reg * (w * (w.clamp_min(1e-12).log())).sum()


        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            # EMA 更新
            self._update_ema()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
            # EMA 更新
            self._update_ema()

        # 返回 GPU 张量，避免每步 CPU 同步；在 epoch 末再做汇总与同步
        return {
            'loss': l.detach(),
            'loss_seg': l_seg.detach(),
            'loss_cls': l_cls.detach()
        }



    
    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            if torch.is_tensor(losses_tr[0]):
                loss_here = torch.stack(losses_tr).float().mean().item()
            else:
                loss_here = np.vstack(losses_tr).mean()

            # --- GAI: 收集 seg 和 cls 训练损失 ---
            losses_seg_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_seg_tr, outputs['loss_seg'])
            if torch.is_tensor(losses_seg_tr[0]):
                loss_seg_here = torch.stack(losses_seg_tr).float().mean().item()
            else:
                loss_seg_here = np.vstack(losses_seg_tr).mean()

            losses_cls_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_cls_tr, outputs['loss_cls'])
            if torch.is_tensor(losses_cls_tr[0]):
                loss_cls_here = torch.stack(losses_cls_tr).float().mean().item()
            else:
                loss_cls_here = np.vstack(losses_cls_tr).mean()
            # --- GAI END ---
        else:
            if torch.is_tensor(outputs['loss']):
                loss_here = outputs['loss'].float().mean().item()
                loss_seg_here = outputs['loss_seg'].float().mean().item()
                loss_cls_here = outputs['loss_cls'].float().mean().item()
            else:
                loss_here = np.mean(outputs['loss'])
                loss_seg_here = np.mean(outputs['loss_seg'])
                loss_cls_here = np.mean(outputs['loss_cls'])
            # --- GAI END ---

        self.logger.log('train_losses', loss_here, self.current_epoch)
        # --- GAI: 记录 seg 和 cls 训练损失 ---
        self.logger.log('train_loss_seg', loss_seg_here, self.current_epoch)
        self.logger.log('train_loss_cls', loss_cls_here, self.current_epoch)
        # W&B: log train metrics
        if self.local_rank == 0 and self.use_wandb and _WANDB_AVAILABLE and self.wandb_run is not None:
            _log = {
                'train/total_loss': float(loss_here),
                'train/seg_loss': float(loss_seg_here),
                'train/cls_loss': float(loss_cls_here),
            }
            try:
                wandb.log(_log, step=self.current_epoch)
            except Exception:
                pass
        

    def on_validation_epoch_start(self):
        self.network.eval()
        if getattr(self, 'ema_model', None) is not None:
            self.ema_model.eval()

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target_seg = batch.get('target', None)
        target_cls_map = batch['class_label']  # voxel map 或 [B]
        keys = batch['keys']                   # ['quiz_0_xxx', 'quiz_1_xxx', ...]

        data = data.to(self.device, non_blocking=True)
        target_cls_map = torch.as_tensor(target_cls_map, device=self.device)

        # 统一得到 case-level 的 target（用于 CE）
        if target_cls_map.ndim == 1:
            target_cls = target_cls_map.long()
        else:
            target_cls = None
            if keys is not None:
                try:
                    parsed = [int(k.split('_')[1]) for k in keys]
                    target_cls = torch.as_tensor(parsed, device=self.device, dtype=torch.long)
                except Exception:
                    target_cls = None
            if target_cls is None:
                tmap = target_cls_map
                if tmap.ndim == 5 and tmap.shape[1] >= 1:
                    tmap = tmap[:, 0]
                elif tmap.ndim != 4:
                    raise RuntimeError(f"Unexpected class_label shape {tuple(tmap.shape)}")
                B = tmap.shape[0]
                flat = tmap.reshape(B, -1).long()
                target_cls = torch.mode(flat, dim=1).values.to(torch.long)

        mode = self.task_mode

        if mode != 'cls_only' and target_seg is not None:
            if isinstance(target_seg, list):
                target_seg = [i.to(self.device, non_blocking=True) for i in target_seg]
            else:
                target_seg = target_seg.to(self.device, non_blocking=True)

        with autocast(self.device.type, dtype=self.amp_dtype) if self.device.type == 'cuda' else dummy_context():
            # 使用 EMA 模型进行验证（若可用）
            net = self.ema_model if getattr(self, 'ema_model', None) is not None else self.network

            # Build ROI for classification to mirror train behavior
            roi_mask_for_cls = None
            if self.task_mode == 'both':
                roi_use_gt_epochs = getattr(self, 'roi_use_gt_epochs', 30)
                if self.current_epoch < roi_use_gt_epochs and target_seg is not None:
                    seg_hires = target_seg[0] if isinstance(target_seg, list) else target_seg
                    if seg_hires.ndim == 4:
                        seg_hires_5d = seg_hires.unsqueeze(1)
                    elif seg_hires.ndim == 5:
                        seg_hires_5d = seg_hires
                    else:
                        raise RuntimeError(f"Unexpected seg_hires ndim={seg_hires.ndim}, expect 4D or 5D.")
                    # 只使用“病灶”作为 ROI：兼容 one-hot 或 labelmap
                    if seg_hires_5d.shape[1] == 1:
                        lesion_val = getattr(self, 'lesion_label_value', 2)
                        roi_mask_for_cls = (seg_hires_5d == lesion_val).float()
                    else:
                        lesion_ch = int(max(0, min(getattr(self.network, 'lesion_channel_idx', 2), seg_hires_5d.shape[1] - 1)))
                        roi_mask_for_cls = (seg_hires_5d[:, lesion_ch:lesion_ch + 1] > 0).float()
                    # match training: light dilation and clamp
                    if getattr(self, 'cls_roi_dilate', True):
                        roi_mask_for_cls = F.max_pool3d(roi_mask_for_cls, kernel_size=3, stride=1, padding=1)
                    roi_mask_for_cls = roi_mask_for_cls.clamp_min(1e-3)
                else:
                    roi_mask_for_cls = None  # use predicted ROI inside model

            output_seg, output_cls = net(data, task_mode=None, roi_mask=roi_mask_for_cls)
            del data
            if mode == 'cls_only':
                l_seg = torch.zeros(1, device=self.device, dtype=torch.float32)
                logits_val = output_cls.float()
                l_cls = self.cls_loss(logits_val, target_cls)
                l = l_cls
            else:
                # seg_only 或 both
                # removed debug stats
                l_seg = self.seg_loss(output_seg, target_seg)
                logits_val = output_cls.float()
                l_cls = self.cls_loss(logits_val, target_cls)
                if mode == 'seg_only':
                    l = l_seg
                else:  # both
                    # 统一与 train_step：Softmax 约束权重
                    w = torch.softmax(self.task_logits, dim=0)
                    # 加性平滑护栏：保证每个任务权重 >= floor，且总和为 1
                    _k = w.shape[0]
                    _floor = min(self.task_weight_floor, 1.0 / _k - 1e-6)
                    w = w * (1.0 - _k * _floor) + _floor
                    l = w[0] * l_seg + w[1] * l_cls
                    if self.task_entropy_reg > 0:
                        l = l + self.task_entropy_reg * (w * (w.clamp_min(1e-12).log())).sum()

        # =========================
        # 分割指标（原逻辑保持不变）
        # =========================
        if mode == 'cls_only':
            if self.label_manager.has_regions:
                n_fg = len(self.label_manager.foreground_regions)
            else:
                n_fg = len(self.label_manager.foreground_labels)
            tp_hard = np.zeros(n_fg, dtype=np.float64)
            fp_hard = np.zeros(n_fg, dtype=np.float64)
            fn_hard = np.zeros(n_fg, dtype=np.float64)
        else:
            # ... 这里保持你原来的 tp/fp/fn 计算逻辑，不改 ...
            if self.enable_deep_supervision:
                output_seg = output_seg[0]
                target_seg = target_seg[0]
            axes = [0] + list(range(2, output_seg.ndim))
            if self.label_manager.has_regions:
                predicted_segmentation_onehot = (torch.sigmoid(output_seg) > 0.5).to(torch.float32)
            else:
                # Safer one-hot via F.one_hot to avoid scatter index issues
                C_here = int(output_seg.shape[1])
                idx = output_seg.argmax(1)  # (B, D, H, W)
                # removed debug stats
                idx = idx.clamp_(0, max(C_here - 1, 0))
                predicted_segmentation_onehot = torch.nn.functional.one_hot(idx, num_classes=C_here)  # (B, D, H, W, C)
                predicted_segmentation_onehot = predicted_segmentation_onehot.movedim(-1, 1).to(torch.float32)

            if self.label_manager.has_ignore_label:
                if not self.label_manager.has_regions:
                    mask = (target_seg != self.label_manager.ignore_label).float()
                    target_seg[target_seg == self.label_manager.ignore_label] = 0
                else:
                    if target_seg.dtype == torch.bool:
                        mask = ~target_seg[:, -1:]
                    else:
                        mask = 1 - target_seg[:, -1:]
                    target_seg = target_seg[:, :-1]
            else:
                # no explicit ignore label in dataset.json. However, padding during cropping may
                # introduce negative labels (e.g., -1). Those must not be used for metrics.
                # We therefore mask out all voxels < 0 and set them to background (0) so that
                # scatter_/one-hot indexing cannot go out of bounds on GPU.
                mask = None
                if not self.label_manager.has_regions:
                    try:
                        neg = target_seg < 0
                        if torch.any(neg):
                            mask = (~neg).float()
                            target_seg[neg] = 0
                    except Exception:
                        # be permissive; if comparison fails due to dtype, just skip
                        mask = None
            # 检测目标中的负值（如 -1），将其并入 mask 中，并把负值写回 0，防止后续 one-hot / 统计越界
            neg = (target_seg < 0)
            if neg.any():
                # 将负值位置纳入屏蔽；与已有 mask 取交集
                add_mask = (~neg).float()
                mask = add_mask if (mask is None) else (mask * add_mask)
                # 避免原张量被其他地方引用导致意外改动，先 clone 再写
                target_seg = target_seg.clone()
                target_seg[neg] = 0

            # 统一为 float，便于 get_tp_fp_fn_tn 的 mask 乘法
            if (mask is not None) and (mask.dtype != torch.float32):
                mask = mask.float()

            tp, fp, fn, _ = get_tp_fp_fn_tn(
                predicted_segmentation_onehot, target_seg, axes=axes, mask=mask
            )

            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()
            if not self.label_manager.has_regions:
                tp_hard = tp_hard[1:]
                fp_hard = fp_hard[1:]
                fn_hard = fn_hard[1:]

        # === Whole-pancreas (label>0) union Dice ===
        if mode == 'cls_only':
            wp_tp = np.array([0.], dtype=np.float64)
            wp_fp = np.array([0.], dtype=np.float64)
            wp_fn = np.array([0.], dtype=np.float64)
        else:
            # 取用于度量的最高分辨率输出/目标
            if self.enable_deep_supervision:
                pred_logits_here = output_seg[0]
                gt_here = target_seg[0]
            else:
                pred_logits_here = output_seg
                gt_here = target_seg

            if self.label_manager.has_regions:
                pred_onehot_here = (torch.sigmoid(pred_logits_here) > 0.5)
                pred_union = pred_onehot_here.any(dim=1, keepdim=True)
                tgt_union = (gt_here > 0).any(dim=1, keepdim=True)
            else:
                # argmax->onehot 已在上面得到 predicted_segmentation_onehot
                pred_union = (predicted_segmentation_onehot[:, 1:, ...].sum(1, keepdim=True) > 0)
                tgt_union = (gt_here != 0)

            if mask is not None:
                mbool = mask.bool() if mask.dtype != torch.bool else mask
                pred_union = pred_union & mbool
                tgt_union = tgt_union & mbool

            # Compute union TP/FP/FN directly using boolean ops to avoid one-hot scatter
            pred_b = pred_union.bool()
            tgt_b = tgt_union.bool()
            axes_wp = [0] + list(range(2, pred_b.ndim))
            tp_wp = (pred_b & tgt_b).sum(dim=axes_wp, keepdim=False).to(torch.float32)
            fp_wp = (pred_b & (~tgt_b)).sum(dim=axes_wp, keepdim=False).to(torch.float32)
            fn_wp = ((~pred_b) & tgt_b).sum(dim=axes_wp, keepdim=False).to(torch.float32)

            # 汇总成标量（numpy）
            wp_tp = tp_wp.detach().cpu().numpy().astype(np.float64).sum(0, keepdims=True)
            wp_fp = fp_wp.detach().cpu().numpy().astype(np.float64).sum(0, keepdims=True)
            wp_fn = fn_wp.detach().cpu().numpy().astype(np.float64).sum(0, keepdims=True)

        # 在返回值里追加：
        return {
            'loss': l.detach().cpu().numpy(),
            'loss_seg': l_seg.detach().cpu().numpy(),
            'loss_cls': l_cls.detach().cpu().numpy(),
            'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard,
            'cls_pred': output_cls.detach().cpu().numpy(),
            'cls_target': target_cls.detach().cpu().numpy(),
            'wp_tp': wp_tp, 'wp_fp': wp_fp, 'wp_fn': wp_fn,   # ★ NEW
        }




    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)

        # --- 分割指标 ---
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            # 损失
            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()

            losses_seg_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_seg_val, outputs_collated['loss_seg'])
            loss_seg_here = np.vstack(losses_seg_val).mean()

            losses_cls_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_cls_val, outputs_collated['loss_cls'])
            loss_cls_here = np.vstack(losses_cls_val).mean()

            # 分类 case-level 预测 & GT（都是 1D int）
            cls_preds_all = [None for _ in range(world_size)]
            dist.all_gather_object(cls_preds_all, outputs_collated['cls_pred'])
            cls_targets_all = [None for _ in range(world_size)]
            dist.all_gather_object(cls_targets_all, outputs_collated['cls_target'])

            cls_preds = np.concatenate(cls_preds_all) if len(cls_preds_all) > 0 else np.array([], dtype=np.int64)
            cls_targets = np.concatenate(cls_targets_all) if len(cls_targets_all) > 0 else np.array([], dtype=np.int64)

        else:
            loss_here = np.mean(outputs_collated['loss'])
            loss_seg_here = np.mean(outputs_collated['loss_seg'])
            loss_cls_here = np.mean(outputs_collated['loss_cls'])

            cls_preds = np.concatenate(outputs_collated['cls_pred']) if len(outputs_collated['cls_pred']) > 0 else np.array([], dtype=np.int64)
            cls_targets = np.concatenate(outputs_collated['cls_target']) if len(outputs_collated['cls_target']) > 0 else np.array([], dtype=np.int64)

        # Dice
        global_dc_per_class = [
            (2 * i / (2 * i + j + k) if (2 * i + j + k) > 0 else 0)
            for i, j, k in zip(tp, fp, fn)
        ]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        
        # --- 5. GAI: 计算分类指标 ---
        # 假设 cls_preds 是 logits/probs，我们需要 argmax 来获取预测类别
        cls_preds_int = np.argmax(cls_preds, axis=1)
        
        # 计算 Macro F1-Score
        macro_f1 = f1_score(cls_targets, cls_preds_int, average='macro', zero_division=0)
        # 计算 Accuracy
        accuracy = accuracy_score(cls_targets, cls_preds_int)
        # 将预测标准化为 1D int，供后续统一使用
        cls_preds = cls_preds_int
        # --- Whole-pancreas union Dice 聚合 ---
        wp_tp = np.sum(outputs_collated['wp_tp'], 0)
        wp_fp = np.sum(outputs_collated['wp_fp'], 0)
        wp_fn = np.sum(outputs_collated['wp_fn'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()
            _tp = [None for _ in range(world_size)]
            _fp = [None for _ in range(world_size)]
            _fn = [None for _ in range(world_size)]
            dist.all_gather_object(_tp, wp_tp)
            dist.all_gather_object(_fp, wp_fp)
            dist.all_gather_object(_fn, wp_fn)
            wp_tp = np.vstack(_tp).sum(0)
            wp_fp = np.vstack(_fp).sum(0)
            wp_fn = np.vstack(_fn).sum(0)

        den_wp = 2 * wp_tp + wp_fp + wp_fn
        wp_dice = float(2 * wp_tp / np.maximum(den_wp, 1e-8)) if den_wp.sum() > 0 else 0.0
        self.logger.log('val_dice_whole_pancreas', wp_dice, self.current_epoch)
        # also track raw counts per epoch for debugging/analysis
        try:
            self.logger.log('val_wp_tp', float(np.array(wp_tp).sum()), self.current_epoch)
            self.logger.log('val_wp_fp', float(np.array(wp_fp).sum()), self.current_epoch)
            self.logger.log('val_wp_fn', float(np.array(wp_fn).sum()), self.current_epoch)
        except Exception:
            # be lenient if shapes are unexpected
            pass

        # 分类指标（case-level, 已经是 int，不需要 argmax）
        if cls_targets.size > 0:
            macro_f1 = f1_score(cls_targets, cls_preds, average='macro', zero_division=0)
            accuracy = accuracy_score(cls_targets, cls_preds)
        else:
            macro_f1 = 0.0
            accuracy = 0.0

        # logging
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)
        self.logger.log('val_loss_seg', loss_seg_here, self.current_epoch)
        self.logger.log('val_loss_cls', loss_cls_here, self.current_epoch)
        self.logger.log('val_macro_f1', macro_f1, self.current_epoch)
        self.logger.log('val_accuracy', accuracy, self.current_epoch)
        # W&B: log validation metrics
        if self.local_rank == 0 and self.use_wandb and _WANDB_AVAILABLE and self.wandb_run is not None:
            log_dict = {
                'val/total_loss': float(loss_here),
                'val/seg_loss': float(loss_seg_here),
                'val/cls_loss': float(loss_cls_here),
                'val/macro_f1': float(macro_f1),
                'val/accuracy': float(accuracy),
                'val/mean_fg_dice': float(mean_fg_dice),
            }
            try:
                for i, v in enumerate(global_dc_per_class):
                    log_dict[f'val/dice_class_{i+1}'] = float(v)
            except Exception:
                pass
            try:
                if 'val_dice_whole_pancreas' in self.logger.my_fantastic_logging:
                    log_dict['val/dice_whole_pancreas'] = float(self.logger.my_fantastic_logging['val_dice_whole_pancreas'][-1])
            except Exception:
                pass
            try:
                wandb.log(log_dict, step=self.current_epoch)
            except Exception:
                pass


    def on_epoch_start(self):
        self.logger.log('epoch_start_timestamps', time(), self.current_epoch)

    def on_epoch_end(self):
        # 看 encoder 里有没有在更新（整块 encoder）
        # self.debug_check_param("encoder")

        # # 看 decoder 里有没有在更新
        # self.debug_check_param("decoder")

        # # 看新的分类头：投影层
        # self.debug_check_param("ct_proj")

        # # 看 Dual-path Transformer 是否在更新
        # self.debug_check_param("dual_block")

        # # 看最终分类输出层
        # self.debug_check_param("cls_out")
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        # --- GAI: 打印所有训练和验证损失 ---
        self.print_to_log_file('train_loss (total)', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('train_loss (seg)', np.round(self.logger.my_fantastic_logging['train_loss_seg'][-1], decimals=4))
        self.print_to_log_file('train_loss (cls)', np.round(self.logger.my_fantastic_logging['train_loss_cls'][-1], decimals=4))
        
        self.print_to_log_file('val_loss (total)', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss (seg)', np.round(self.logger.my_fantastic_logging['val_loss_seg'][-1], decimals=4))
        self.print_to_log_file('val_loss (cls)', np.round(self.logger.my_fantastic_logging['val_loss_cls'][-1], decimals=4))

        # --- GAI: 打印你关心的特定分割指标 ---
        dice_scores = self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]
        self.print_to_log_file('Pseudo dice (per class)', [np.round(i, decimals=4) for i in dice_scores])
        
        if 'val_dice_whole_pancreas' in self.logger.my_fantastic_logging:
            self.print_to_log_file('Whole-pancreas Dice', 
                np.round(self.logger.my_fantastic_logging['val_dice_whole_pancreas'][-1], 4))

        
        # 计算 Whole Pancreas (Label 1 + Label 2)
        # 注意：这需要修改 validation_step 来计算合并标签的 TP/FP/FN。
        # 目前，我们只能打印单类的指标。
        # (打印单类指标)
        if len(dice_scores) > 0:
            self.print_to_log_file(f'  -> Pancreas (Label 1) Dice: {np.round(dice_scores[0], decimals=4)}')
        if len(dice_scores) > 1:
            self.print_to_log_file(f'  -> Lesion (Label 2) Dice: {np.round(dice_scores[1], decimals=4)}')

        # --- GAI: 打印分类指标（macro-average F1 与 accuracy） ---
        if 'val_macro_f1' in self.logger.my_fantastic_logging and \
           len(self.logger.my_fantastic_logging['val_macro_f1']) > 0:
            self.print_to_log_file('Classification macro-average F1', 
                                   np.round(self.logger.my_fantastic_logging['val_macro_f1'][-1], 4))
        if 'val_accuracy' in self.logger.my_fantastic_logging and \
           len(self.logger.my_fantastic_logging['val_accuracy']) > 0:
            self.print_to_log_file('Classification accuracy', 
                                   np.round(self.logger.my_fantastic_logging['val_accuracy'][-1], 4))

        # 可选：打印 Whole‑pancreas 的 TP/FP/FN 计数（若已记录）
        if 'val_wp_tp' in self.logger.my_fantastic_logging and len(self.logger.my_fantastic_logging['val_wp_tp']) > 0:
            self.print_to_log_file('Whole-pancreas TP', self.logger.my_fantastic_logging['val_wp_tp'][-1])
        if 'val_wp_fp' in self.logger.my_fantastic_logging and len(self.logger.my_fantastic_logging['val_wp_fp']) > 0:
            self.print_to_log_file('Whole-pancreas FP', self.logger.my_fantastic_logging['val_wp_fp'][-1])
        if 'val_wp_fn' in self.logger.my_fantastic_logging and len(self.logger.my_fantastic_logging['val_wp_fn']) > 0:
            self.print_to_log_file('Whole-pancreas FN', self.logger.my_fantastic_logging['val_wp_fn'][-1])
        
        # 打印 Mean Foreground Dice
        self.print_to_log_file('Mean Foreground Dice', np.round(self.logger.my_fantastic_logging['mean_fg_dice'][-1], decimals=4))

        # 分类详细输出已在上方打印（macro-F1 与 accuracy）

        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        # GAI: 我们现在使用 'mean_fg_dice' 来判断最佳模型，而不是 ema_fg_dice
        current_best = self.logger.my_fantastic_logging['mean_fg_dice'][-1]
        if self._best_ema is None or current_best > self._best_ema:
            self._best_ema = current_best
            self.print_to_log_file(f"Yayy! New best Mean pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        # handle 'best macro-f1' checkpointing
        try:
            if 'val_macro_f1' in self.logger.my_fantastic_logging and \
               len(self.logger.my_fantastic_logging['val_macro_f1']) > 0:
                current_f1 = self.logger.my_fantastic_logging['val_macro_f1'][-1]
                if (self._best_macro_f1 is None) or (current_f1 > self._best_macro_f1):
                    self._best_macro_f1 = current_f1
                    self.print_to_log_file(f"Yayy! New best Macro F1: {np.round(self._best_macro_f1, decimals=4)}")
                    self.save_checkpoint(join(self.output_folder, 'checkpoint_best_macro_f1.pth'))
        except Exception as _e:
            # be lenient: macro-f1 may be missing early on
            pass

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    '_best_macro_f1': self._best_macro_f1,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                }
                # 若存在 EMA，附带保存（推理端将优先使用）
                if getattr(self, 'ema_model', None) is not None:
                    try:
                        checkpoint['ema_network_weights'] = self.ema_model.state_dict()
                    except Exception:
                        pass
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self._best_macro_f1 = checkpoint.get('_best_macro_f1', self._best_macro_f1)
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

        # 如果存在 EMA 权重，则尝试加载到本地 EMA 模型
        if 'ema_network_weights' in checkpoint:
            try:
                self._init_ema()
                self.ema_model.load_state_dict(checkpoint['ema_network_weights'])
            except Exception:
                pass

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()
        net_for_val = self.ema_model if getattr(self, 'ema_model', None) is not None else self.network
        # 关闭 EMA 模型上的深监督
        try:
            mod = net_for_val.module if isinstance(net_for_val, DDP) else net_for_val
            if isinstance(mod, OptimizedModule):
                mod = mod._orig_mod
            if hasattr(mod, 'decoder'):
                mod.decoder.deep_supervision = False
            if hasattr(mod, 'deep_supervision'):
                mod.deep_supervision = False
        except Exception:
            pass

        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                                   "encounter crashes in validation then this is because torch.compile forgets "
                                   "to trigger a recompilation of the model with deep supervision disabled. "
                                   "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                                   "validation with --val (exactly the same as before) and then it will work. "
                                   "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                                   "forward pass (where compile is triggered) already has deep supervision disabled. "
                                   "This is exactly what we need in perform_actual_validation")

        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                        perform_everything_on_device=True, device=self.device, verbose=False,
                                        verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(net_for_val, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            _, val_keys = self.do_split()
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1
                val_keys = val_keys[self.local_rank:: dist.get_world_size()]

            dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                             folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []
            
            # --- GAI: 为分类结果创建一个字典 ---
            all_classification_results = {}
            # --- GAI END ---

            for i, k in enumerate(dataset_val.identifiers):
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                           allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                                   allowed_num_queued=2)

                self.print_to_log_file(f"predicting {k}")
                data, _, seg_prev, properties = dataset_val.load_case(k)

                data = data[:]

                if self.is_cascaded:
                    seg_prev = seg_prev[:]
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg_prev, self.label_manager.foreground_labels,
                                                                         output_dtype=data.dtype)))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
                output_filename_truncated = join(validation_output_folder, k)

                # --- GAI: 捕获 (seg, cls) 两个输出 ---
                prediction_seg, prediction_cls = predictor.predict_sliding_window_return_logits(data)

                # (1) 处理分割 (移动到 CPU)
                prediction_seg = prediction_seg.cpu()

                # (2) 处理分类 (移动到 CPU, 转换为列表, 并存储)
                # 我们将病例 ID (k) 作为键
                all_classification_results[k] = prediction_cls.cpu().numpy().tolist()
                # --- GAI END ---

                # this needs to go into background processes
                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            # --- GAI: 传递 prediction_seg 而不是 prediction ---
                            (prediction_seg, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, save_probabilities),
                            # --- GAI END ---
                        )
                    )
                )

                # if needed, export the softmax prediction for the next stage
                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
                                                            next_stage_config_manager.data_identifier)
                        dataset_class = infer_dataset_class(expected_preprocessed_folder)

                        try:
                            tmp = dataset_class(expected_preprocessed_folder, [k])
                            d, _, _, _ = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                                f"Run the preprocessing for this configuration first!")
                            continue

                        target_shape = d.shape[1:]
                        output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                        output_file_truncated = join(output_folder, k)

                        results.append(segmentation_export_pool.starmap_async(
                            resample_and_save, (
                                # --- GAI: 传递 prediction_seg 而不是 prediction ---
                                (prediction_seg, target_shape, output_file_truncated, self.plans_manager,
                                 self.configuration_manager,
                                 properties,
                                 self.dataset_json,
                                 default_num_processes,
                                 dataset_class),
                                # --- GAI END ---
                            )
                        ))
                
                if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                    dist.barrier()

            _ = [r.get() for r in results]


            if self.local_rank == 0 and all_classification_results:
                self.print_to_log_file("Saving classification results...")
                save_json(all_classification_results,
                          join(validation_output_folder, 'classification_results.json'))


            if self.is_ddp:
                dist.barrier()

            if self.local_rank == 0:
                metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                    validation_output_folder,
                                                    join(validation_output_folder, 'summary.json'),
                                                    self.plans_manager.image_reader_writer_class(),
                                                    self.dataset_json["file_ending"],
                                                    self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                    self.label_manager.foreground_labels,
                                                    self.label_manager.ignore_label, chill=True,
                                                    num_processes=default_num_processes * dist.get_world_size() if
                                                    self.is_ddp else default_num_processes)
                self.print_to_log_file("Validation complete", also_print_to_console=True)
                self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                       also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()
    def debug_check_param(self, layer_keyword: str = "classification_head"):
        """
        打印某一层参数的统计信息，并检查是否在更新 & 是否出现 NaN.
        layer_keyword: 在 parameter name 里查找的关键字，比如 "classification_head" 或 "cls_adapter"
        """
        # 处理 DDP 情况
        if self.is_ddp:
            net = self.network.module
        else:
            net = self.network

        # 找到第一个名字里包含 layer_keyword 的参数
        for name, p in net.named_parameters():
            if layer_keyword in name and p.requires_grad:
                w = p.data
                # 1) NaN 检查
                has_nan = torch.isnan(w).any().item()

                # 2) 基本统计量
                w_mean = w.mean().item()
                w_std  = w.std().item()
                w_min  = w.min().item()
                w_max  = w.max().item()

                # 3) 更新幅度检查：和上一次保存的权重对比
                key = f"_debug_prev_{layer_keyword}"
                prev = getattr(self, key, None)
                if prev is None:
                    # 第一次调用：只存一份，不比较
                    setattr(self, key, w.detach().clone())
                    delta_mean = 0.0
                    delta_max  = 0.0
                else:
                    diff = (w - prev).abs()
                    delta_mean = diff.mean().item()
                    delta_max  = diff.max().item()
                    # 更新缓存
                    setattr(self, key, w.detach().clone())
                g = p.grad
                if g is not None:
                    g_has_nan = torch.isnan(g).any().item()
                    g_mean = g.mean().item()
                    g_std  = g.std().item()
                else:
                    g_has_nan = False
                    g_mean = 0.0
                    g_std  = 0.0

                self.print_to_log_file(
                    f"[DEBUG] Layer '{name}': "
                    f"w_mean={w_mean:.4e}, w_std={w_std:.4e}, "
                    f"w_min={w_min:.4e}, w_max={w_max:.4e}, "
                    f"w_has_nan={has_nan}, "
                    f"g_mean={g_mean:.4e}, g_std={g_std:.4e}, g_has_nan={g_has_nan}, "
                    f"delta_mean={delta_mean:.4e}, delta_max={delta_max:.4e}"
                )
                self.print_to_log_file(
                    f"[DEBUG] Layer '{name}': "
                    f"mean={w_mean:.4e}, std={w_std:.4e}, "
                    f"min={w_min:.4e}, max={w_max:.4e}, "
                    f"has_nan={has_nan}, "
                    f"delta_mean={delta_mean:.4e}, delta_max={delta_max:.4e}"
                )
                break
        else:
            self.print_to_log_file(f"[DEBUG] No parameter found with keyword '{layer_keyword}'")

    def run_training(self):
        self.on_train_start()
        # debug: 打印训练集中每个分类标签的样本数量
        # from collections import Counter
        # labels = [int(k.split('_')[1]) for k in self.tr_keys]
        # print('label counts:', Counter(labels))


        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()


# Plan
# nnUNetv2_plan_and_preprocess -d 002 -pl nnUNetPlannerResEncM
# train
# nnUNetv2_train 002 3d_fullres 4 -p nnUNetResEncUNetMPlans 
# train using all data 
# nnUNetv2_train 002 3d_fullres 5 -p nnUNetResEncUNetMPlans
# train cls
# nnUNetv2_train 002 3d_fullres 5 -p nnUNetResEncUNetMPlans -pretrained_weights F:\Programming\JupyterWorkDir\labquiz\ML-Quiz-3DMedImg\bestsig\20251107_best3.pth

# predict
# nnUNetv2_predict -i F:\Programming\JupyterWorkDir\labquiz\ML-Quiz-3DMedImg\validation\img -o F:\Programming\JupyterWorkDir\labquiz\ML-Quiz-3DMedImg\validation\prediction -d 002 -c 3d_fullres -p nnUNetResEncUNetMPlans -f 5 -chk  checkpoint_best_macro_f1.pth
# nnUNetv2_predict -i F:\Programming\JupyterWorkDir\labquiz\ML-Quiz-3DMedImg\validation\img -o F:\Programming\JupyterWorkDir\labquiz\ML-Quiz-3DMedImg\validation\prediction -d 002 -c 3d_fullres -p nnUNetResEncUNetMPlans -f 5 -chk  checkpoint_best.pth
# nnUNetv2_predict -i F:\Programming\JupyterWorkDir\labquiz\ML-Quiz-3DMedImg\validation\img -o F:\Programming\JupyterWorkDir\labquiz\ML-Quiz-3DMedImg\validation\prediction -d 002 -c 3d_fullres -p nnUNetResEncUNetMPlans -f 5 -chk  checkpoint_best.pth -step_size 0.2
# submission
# nnUNetv2_predict -i F:\Programming\JupyterWorkDir\labquiz\ML-Quiz-3DMedImg\test -o F:\Programming\JupyterWorkDir\labquiz\ML-Quiz-3DMedImg\Baidu_Li_Results -d 002 -c 3d_fullres -p nnUNetResEncUNetMPlans -f 5 -chk  checkpoint_best_macro_f1.pth

if __name__ == "__main__":
    import torch
    import numpy as np
    from torch import nn
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.ResNet_MTL_nnUNet import ResNet_MTL_nnUNet  # ⚠️ 确保你的模型类路径正确（根据项目结构调整）
    from dynamic_network_architectures.building_blocks.residual import BasicBlockD


    print("🚀 Debug training using ResNet_MTL_nnUNet + nnUNetTrainer ...")

    # ===============================================================
    # 1️⃣ 初始化模型（参数配置与你在主脚本中一致）
    # ===============================================================
    unet_config = {
        "input_channels": 1,
        "n_stages": 6,
        "features_per_stage": (32, 64, 128, 256, 320, 320),
        "conv_op": nn.Conv3d,
        "kernel_sizes": ((3, 3, 3),) * 6,
        "strides": ((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        "n_blocks_per_stage": (2, 2, 2, 2, 2, 2),
        "num_classes": 2,  # segmentation
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
        "cls_num_classes": 3,
        "cls_query_num": 8,
        "cls_dropout": 0.1,
        "use_cross_attention": True,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet_MTL_nnUNet(**unet_config).to(device)

    # ===============================================================
    # 2️⃣ 模拟 nnUNetTrainer 环境
    # ===============================================================
    from nnUNetTrainer import nnUNetTrainer  # 确认类名正确
    trainer = nnUNetTrainer.__new__(nnUNetTrainer)  # 不调用 init
    trainer.device = device
    trainer.network = model
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    trainer.grad_scaler = None
    trainer.lambda_seg = 1.0
    trainer.lambda_cls = 1.0
    trainer.seg_loss = nn.CrossEntropyLoss()
    trainer.cls_loss = nn.CrossEntropyLoss()



    # ===============================================================
    # 3️⃣ 构造假数据
    # ===============================================================
    B, C, D, H, W = 2, 1, 64, 64, 64
    dummy_data = torch.randn(B, C, D, H, W).to(device)
    dummy_seg = torch.randint(0, 2, (B, D, H, W)).to(device)
    dummy_cls = torch.randint(0, 3, (B,)).to(device)

    batch = {
        'data': dummy_data,
        'target': dummy_seg,
        'class_label': dummy_cls
    }

    # ===============================================================
    # 4️⃣ 调用 train_step（调试模式）
    # ===============================================================
    print("\n===== Running one debug train step =====")
    result = trainer.train_step(batch)

    print("\n✅ Train step finished.")
    print(f"Loss_total={result['loss']:.4f}, "
          f"Loss_seg={result['loss_seg']:.4f}, "
          f"Loss_cls={result['loss_cls']:.4f}")

    # ===============================================================
    # 5️⃣ 打印分类头梯度
    # ===============================================================
    print("\n--- Classification Head Gradient Debug ---")
    found_grad = False
    for name, p in trainer.network.named_parameters():
        if 'classification_head' in name or 'cls' in name:
            if p.grad is None:
                print(f"[!] No grad for {name}")
            else:
                print(f"grad[{name}] mean={p.grad.abs().mean().item():.6f}")
                found_grad = True
    if not found_grad:
        print("⚠️ No classification head gradients detected!")

    print("\n🎯 Debug complete — if you see nonzero grad above, classification head is learning!")
