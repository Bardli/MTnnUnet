import inspect
import itertools
import multiprocessing
import os
from copy import deepcopy
from queue import Queue
from threading import Thread
from time import sleep, time
from typing import Tuple, Union, List, Optional

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy, preprocessing_iterator_fromfiles, \
    preprocessing_iterator_fromnpy
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder


class nnUNetPredictor(object):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True,
                 # --- GAI: 新增分类相关参数 ---
                 cls_extended_tta: bool = True,
                 ):
        self.verbose = verbose
        self.verbose_preprocessing = verbose_preprocessing
        self.allow_tqdm = allow_tqdm

        self.plans_manager, self.configuration_manager, self.list_of_parameters, self.network, self.dataset_json, \
        self.trainer_name, self.allowed_mirroring_axes, self.label_manager = None, None, None, None, None, None, None, None

        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        else:
            print(f'perform_everything_on_device=True is only supported for cuda devices! Setting this to False')
            perform_everything_on_device = False
        self.device = device
        self.perform_everything_on_device = perform_everything_on_device
        # --- GAI: 分类 TTA 控制 ---
        self.cls_extended_tta = cls_extended_tta

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'), weights_only=False)
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None
            # 优先使用 EMA 权重（若可用）
            net_state = checkpoint.get('ema_network_weights', checkpoint['network_weights'])
            parameters.append(net_state)

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                    trainer_name, 'nnunetv2.training.nnUNetTrainer')



        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. '
                               f'Please place it there (in any .py file)!')
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            task_mode='both',
            enable_deep_supervision=False
        )


        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters

        # initialize network with first set of parameters, also see https://github.com/MIC-DKFZ/nnUNet/issues/2520
        network.load_state_dict(parameters[0])

        self.network = network

        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    def manual_initialization(self, network: nn.Module, plans_manager: PlansManager,
                              configuration_manager: ConfigurationManager, parameters: Optional[List[dict]],
                              dataset_json: dict, trainer_name: str,
                              inference_allowed_mirroring_axes: Optional[Tuple[int, ...]]):
        """
        This is used by the nnUNetTrainer to initialize nnUNetPredictor for the final validation
        """
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        allow_compile = True
        allow_compile = allow_compile and ('nnUNet_compile' in os.environ.keys()) and (
                    os.environ['nnUNet_compile'].lower() in ('true', '1', 't'))
        allow_compile = allow_compile and not isinstance(self.network, OptimizedModule)
        if isinstance(self.network, DistributedDataParallel):
            allow_compile = allow_compile and isinstance(self.network.module, OptimizedModule)
        if allow_compile:
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    @staticmethod
    def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
        print('use_folds is None, attempting to auto detect available folds')
        fold_folders = subdirs(model_training_output_dir, prefix='fold_', join=False)
        fold_folders = [i for i in fold_folders if i != 'fold_all']
        fold_folders = [i for i in fold_folders if isfile(join(model_training_output_dir, i, checkpoint_name))]
        use_folds = [int(i.split('_')[-1]) for i in fold_folders]
        print(f'found the following folds: {use_folds}')
        return use_folds

    def _manage_input_and_output_lists(self, list_of_lists_or_source_folder: Union[str, List[List[str]]],
                                       output_folder_or_list_of_truncated_output_files: Union[None, str, List[str]],
                                       folder_with_segs_from_prev_stage: str = None,
                                       overwrite: bool = True,
                                       part_id: int = 0,
                                       num_parts: int = 1,
                                       save_probabilities: bool = False):
        if isinstance(list_of_lists_or_source_folder, str):
            list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(list_of_lists_or_source_folder,
                                                                                       self.dataset_json['file_ending'])
        print(f'There are {len(list_of_lists_or_source_folder)} cases in the source folder')
        list_of_lists_or_source_folder = list_of_lists_or_source_folder[part_id::num_parts]
        caseids = [os.path.basename(i[0])[:-(len(self.dataset_json['file_ending']) + 5)] for i in
                   list_of_lists_or_source_folder]
        print(
            f'I am processing {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)')
        print(f'There are {len(caseids)} cases that I would like to predict')

        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_filename_truncated = [join(output_folder_or_list_of_truncated_output_files, i) for i in caseids]
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_filename_truncated = output_folder_or_list_of_truncated_output_files[part_id::num_parts]
        else:
            output_filename_truncated = None
        seg_from_prev_stage_files = [join(folder_with_segs_from_prev_stage, i + self.dataset_json['file_ending']) if
                                     folder_with_segs_from_prev_stage is not None else None for i in caseids]
        # remove already predicted files from the lists
        if not overwrite and output_filename_truncated is not None:
            tmp = [isfile(i + self.dataset_json['file_ending']) for i in output_filename_truncated]
            if save_probabilities:
                tmp2 = [isfile(i + '.npz') for i in output_filename_truncated]
                tmp = [i and j for i, j in zip(tmp, tmp2)]
            not_existing_indices = [i for i, j in enumerate(tmp) if not j]

            output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
            list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in not_existing_indices]
            seg_from_prev_stage_files = [seg_from_prev_stage_files[i] for i in not_existing_indices]
            print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
                  f'That\'s {len(not_existing_indices)} cases.')
        return list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files

    def predict_from_files(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           num_processes_preprocessing: int = default_num_processes,
                           num_processes_segmentation_export: int = default_num_processes,
                           folder_with_segs_from_prev_stage: str = None,
                           num_parts: int = 1,
                           part_id: int = 0):
        """
        This is nnU-Net's default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """
        assert part_id <= num_parts, ("Part ID must be smaller than num_parts. Remember that we start counting with 0. "
                                      "So if there are 3 parts then valid part IDs are 0, 1, 2")
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        else:
            output_folder = None

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(
                my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
            recursive_fix_for_json_export(my_init_kwargs)
            maybe_mkdir_p(output_folder)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

            # we need these two if we want to do things with the predictions like for example apply postprocessing
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        #######################

        # check if we need a prediction from the previous stage
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
                f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
                f' they are located via folder_with_segs_from_prev_stage'

        # sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(list_of_lists_or_source_folder,
                                                                                 seg_from_prev_stage_files,
                                                                                 output_filename_truncated,
                                                                                 num_processes_preprocessing)

        return self.predict_from_data_iterator(data_iterator, save_probabilities, num_processes_segmentation_export,
                                                 output_folder)

    def _internal_get_data_iterator_from_lists_of_filenames(self,
                                                            input_list_of_lists: List[List[str]],
                                                            seg_from_prev_stage_files: Union[List[str], None],
                                                            output_filenames_truncated: Union[List[str], None],
                                                            num_processes: int):
        return preprocessing_iterator_fromfiles(input_list_of_lists, seg_from_prev_stage_files,
                                                output_filenames_truncated, self.plans_manager, self.dataset_json,
                                                self.configuration_manager, num_processes, self.device.type == 'cuda',
                                                self.verbose_preprocessing)
        # preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose_preprocessing)
        # # hijack batchgenerators, yo
        # # we use the multiprocessing of the batchgenerators dataloader to handle all the background worker stuff. This
        # # way we don't have to reinvent the wheel here.
        # num_processes = max(1, min(num_processes, len(input_list_of_lists)))
        # ppa = PreprocessAdapter(input_list_of_lists, seg_from_prev_stage_files, preprocessor,
        #                         output_filenames_truncated, self.plans_manager, self.dataset_json,
        #                         self.configuration_manager, num_processes)
        # if num_processes == 0:
        #     mta = SingleThreadedAugmenter(ppa, None)
        # else:
        #     mta = MultiThreadedAugmenter(ppa, None, num_processes, 1, None, pin_memory=pin_memory)
        # return mta

    def get_data_iterator_from_raw_npy_data(self,
                                            image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                            segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None,
                                                                                                        np.ndarray,
                                                                                                        List[
                                                                                                            np.ndarray]],
                                            properties_or_list_of_properties: Union[dict, List[dict]],
                                            truncated_ofname: Union[str, List[str], None],
                                            num_processes: int = 3):

        list_of_images = [image_or_list_of_images] if not isinstance(image_or_list_of_images, list) else \
            image_or_list_of_images

        if isinstance(segs_from_prev_stage_or_list_of_segs_from_prev_stage, np.ndarray):
            segs_from_prev_stage_or_list_of_segs_from_prev_stage = [
                segs_from_prev_stage_or_list_of_segs_from_prev_stage]

        if isinstance(truncated_ofname, str):
            truncated_ofname = [truncated_ofname]

        if isinstance(properties_or_list_of_properties, dict):
            properties_or_list_of_properties = [properties_or_list_of_properties]

        num_processes = min(num_processes, len(list_of_images))
        pp = preprocessing_iterator_fromnpy(
            list_of_images,
            segs_from_prev_stage_or_list_of_segs_from_prev_stage,
            properties_or_list_of_properties,
            truncated_ofname,
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_processes,
            self.device.type == 'cuda',
            self.verbose_preprocessing
        )

        return pp

    def predict_from_list_of_npy_arrays(self,
                                        image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                        segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None,
                                                                                                    np.ndarray,
                                                                                                    List[
                                                                                                        np.ndarray]],
                                        properties_or_list_of_properties: Union[dict, List[dict]],
                                        truncated_ofname: Union[str, List[str], None],
                                        num_processes: int = 3,
                                        save_probabilities: bool = False,
                                        num_processes_segmentation_export: int = default_num_processes):
        iterator = self.get_data_iterator_from_raw_npy_data(image_or_list_of_images,
                                                            segs_from_prev_stage_or_list_of_segs_from_prev_stage,
                                                            properties_or_list_of_properties,
                                                            truncated_ofname,
                                                            num_processes)
        return self.predict_from_data_iterator(iterator, save_probabilities, num_processes_segmentation_export)

    def predict_from_data_iterator(self,
                                     data_iterator,
                                     save_probabilities: bool = False,
                                     num_processes_segmentation_export: int = default_num_processes,
                                     output_folder: str = None): # <-- GAI: 添加第 5 个参数 (output_folder)
        """
        each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
        If 'ofile' is None, the result will be returned instead of written to a file
        """
        
        # --- GAI: 为分类结果添加两个容器 ---
        all_classification_results = {}
        subtype_csv_rows = []
        # --- GAI 结束 ---
        
        start_time = time()

        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                
                # --- GAI: 如果 output_folder 未传入，尝试从 ofile 推断 ---
                # (这确保了旧的 predict_from_data_iterator 调用仍能工作)
                if output_folder is None and ofile is not None:
                    output_folder = os.path.dirname(ofile)
                # --- GAI 结束 ---
                    
                if ofile is not None:
                    print(f'\nPredicting {os.path.basename(ofile)}:')
                else:
                    print(f'\nPredicting image of shape {data.shape}:')

                print(f'perform_everything_on_device: {self.perform_everything_on_device}')

                properties = preprocessed['data_properties']

                # ... (check_workers_alive_and_busy 逻辑保持不变) ...
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                # --- GAI: 获取 (seg, cls) 两个预测结果（第一阶段：分割+默认TTA分类） ---
                prediction_seg, prediction_cls = self.predict_logits_from_preprocessed_data(data)
                prediction_seg_np = prediction_seg.cpu().detach().numpy()
                prediction_cls_np = prediction_cls.cpu().detach().numpy()
                # --- GAI 结束 ---

                if ofile is not None:
                    print('sending off prediction to background worker for resampling and export')
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            ((prediction_seg_np, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, ofile, save_probabilities),)
                        )
                    )
                    # --- GAI: 保存 cls 结果 ---
                    case_id = os.path.basename(ofile)
                    all_classification_results[case_id] = prediction_cls_np
                    # 同时基于第一阶段分类结果写入 CSV（整例 softmax argmax）
                    probs = torch.softmax(prediction_cls, dim=1).cpu().numpy()[0]
                    pred_label = int(np.argmax(probs))
                    subtype_csv_rows.append({'Names': case_id, 'Subtype': pred_label})
                    # --- GAI 结束 ---
                else:
                    print('sending off prediction to background worker for resampling')
                    r.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape, (
                                (prediction_seg_np, self.plans_manager,
                                 self.configuration_manager, self.label_manager,
                                 properties,
                                 save_probabilities),)
                        )
                    )
                if ofile is not None:
                    print(f'done with {os.path.basename(ofile)}')
                else:
                    print(f'\nDone with image of shape {data.shape}:')

                # 单阶段推理：跳过 ROI 驱动的第二阶段分类细化
                continue

                # --- GAI: 第二阶段 - 基于分割ROI的分类TTA软投票（只做分类，不影响分割） ---
                try:
                    final_cls_logits = self._classify_with_roi_tta(
                        data, prediction_seg,
                        use_extended_tta=self.cls_extended_tta
                    )
                    final_cls_np = final_cls_logits.cpu().numpy()

                    # 覆盖 JSON 中此病例的分类结果，并收集到 CSV 行
                    if ofile is not None:
                        all_classification_results[case_id] = final_cls_np

                        # 导出为 CSV 的一行：Names + Subtype（将向量变成单个数字标签）
                        probs = torch.softmax(final_cls_logits, dim=1).cpu().numpy()[0]
                        pred_label = int(np.argmax(probs))
                        subtype_csv_rows.append({'Names': case_id, 'Subtype': pred_label})
                except Exception as e:
                    print(f'[WARN] ROI-based subtype classification failed for {ofile}: {e}')
            ret = [i.get()[0] for i in r]

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        # --- GAI: 将所有 CLS 结果保存到 JSON 文件 ---
        # (现在使用传入的 output_folder)
        if all_classification_results and output_folder is not None:
            print(f"Saving classification results to {join(output_folder, 'classification_results.json')}")
            cls_results_exportable = {k: v.tolist() for k, v in all_classification_results.items()}
            save_json(cls_results_exportable, join(output_folder, 'classification_results.json'))

        # --- GAI: 导出 subtype_results.csv ---
        if subtype_csv_rows and output_folder is not None:
            import csv
            csv_path = join(output_folder, 'subtype_results.csv')
            print(f"Saving subtype results CSV to {csv_path}")
            # 固定表头为 Names, Subtype
            fieldnames = ['Names', 'Subtype']
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in subtype_csv_rows:
                    writer.writerow(r)
        # --- GAI 结束 ---

        total_time = time() - start_time
        print(f"Total inference time (all cases, multiprocessing): {total_time:.2f} seconds")

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return ret

    def predict_single_npy_array(self, input_image: np.ndarray, image_properties: dict,
                                 segmentation_previous_stage: np.ndarray = None,
                                 output_file_truncated: str = None,
                                 save_or_return_probabilities: bool = False):
        """
        WARNING: SLOW. ONLY USE THIS IF YOU CANNOT GIVE NNUNET MULTIPLE IMAGES AT ONCE FOR SOME REASON.


        input_image: Make sure to load the image in the way nnU-Net expects! nnU-Net is trained on a certain axis
                     ordering which cannot be disturbed in inference,
                     otherwise you will get bad results. The easiest way to achieve that is to use the same I/O class
                     for loading images as was used during nnU-Net preprocessing! You can find that class in your
                     plans.json file under the key "image_reader_writer". If you decide to freestyle, know that the
                     default axis ordering for medical images is the one from SimpleITK. If you load with nibabel,
                     you need to transpose your axes AND your spacing from [x,y,z] to [z,y,x]!
        image_properties must only have a 'spacing' key!
        """
        ppa = PreprocessAdapterFromNpy([input_image], [segmentation_previous_stage], [image_properties],
                                       [output_file_truncated],
                                       self.plans_manager, self.dataset_json, self.configuration_manager,
                                       num_threads_in_multithreaded=1, verbose=self.verbose)
        if self.verbose:
            print('preprocessing')
        dct = next(ppa)

        if self.verbose:
            print('predicting')
        predicted_logits = self.predict_logits_from_preprocessed_data(dct['data']).cpu()

        if self.verbose:
            print('resampling to original shape')
        if output_file_truncated is not None:
            export_prediction_from_logits(predicted_logits, dct['data_properties'], self.configuration_manager,
                                          self.plans_manager, self.dataset_json, output_file_truncated,
                                          save_or_return_probabilities)
        else:
            ret = convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits, self.plans_manager,
                                                                              self.configuration_manager,
                                                                              self.label_manager,
                                                                              dct['data_properties'],
                                                                              return_probabilities=
                                                                              save_or_return_probabilities)
            if save_or_return_probabilities:
                return ret[0], ret[1]
            else:
                return ret

    @torch.inference_mode()
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        IMPORTANT! IF YOU ARE RUNNING THE CASCADE, THE SEGMENTATION FROM THE PREVIOUS STAGE MUST ALREADY BE STACKED ON
        TOP OF THE IMAGE AS ONE-HOT REPRESENTATION! SEE PreprocessAdapter ON HOW THIS SHOULD BE DONE!

        RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.
        SEE convert_predicted_logits_to_segmentation_with_correct_shape
        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)

        # ------- 情况 1：单折或者没有 parameters，直接用当前 self.network -------
        if not self.list_of_parameters or len(self.list_of_parameters) == 1:
            # initialize_from_trained_model_folder 里已经 load 过一次 parameters[0]
            # manual_initialization 的场景下，trainer 也已经把权重放进 self.network 了
            prediction_seg, prediction_cls = self.predict_sliding_window_return_logits(data)
            prediction_seg = prediction_seg.to('cpu')
            prediction_cls = prediction_cls.to('cpu')

        # ------- 情况 2：真多折 ensemble，保留原有逻辑 -------
        else:
            prediction_seg = None
            prediction_cls = None

            for params in self.list_of_parameters:
                # 这里只在需要 ensemble 的时候才切换权重
                if not isinstance(self.network, OptimizedModule):
                    self.network.load_state_dict(params)
                else:
                    self.network._orig_mod.load_state_dict(params)

                seg, cls = self.predict_sliding_window_return_logits(data)
                seg = seg.to('cpu')
                cls = cls.to('cpu')

                if prediction_seg is None:
                    prediction_seg, prediction_cls = seg, cls
                else:
                    prediction_seg += seg
                    prediction_cls += cls

            # 多折平均
            prediction_seg /= len(self.list_of_parameters)
            prediction_cls /= len(self.list_of_parameters)

        if self.verbose:
            print('Prediction done')

        torch.set_num_threads(n_threads)
        return prediction_seg, prediction_cls

    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]):
        slicers = []
        if len(self.configuration_manager.patch_size) < len(image_size):
            assert len(self.configuration_manager.patch_size) == len(
                image_size) - 1, 'if tile_size has less entries than image_size, ' \
                                 'len(tile_size) ' \
                                 'must be one shorter than len(image_size) ' \
                                 '(only dimension ' \
                                 'discrepancy of 1 allowed).'
            steps = compute_steps_for_sliding_window(image_size[1:], self.configuration_manager.patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is'
                                   f' {image_size}, tile_size {self.configuration_manager.patch_size}, '
                                   f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple([slice(None), d, *[slice(si, si + ti) for si, ti in
                                                     zip((sx, sy), self.configuration_manager.patch_size)]]))
        else:
            steps = compute_steps_for_sliding_window(image_size, self.configuration_manager.patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(
                f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {self.configuration_manager.patch_size}, '
                f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                  zip((sx, sy, sz), self.configuration_manager.patch_size)]]))
        return slicers

    # @torch.inference_mode()
    # def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
    #     mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
    #     prediction = self.network(x)

    #     if mirror_axes is not None:
    #         # check for invalid numbers in mirror_axes
    #         # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
    #         assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

    #         mirror_axes = [m + 2 for m in mirror_axes]
    #         axes_combinations = [
    #             c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
    #         ]
    #         for axes in axes_combinations:
    #             prediction += torch.flip(self.network(torch.flip(x, axes)), axes)
    #         prediction /= (len(axes_combinations) + 1)
    #     return prediction
    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        适配多任务网络:
        - self.network(x) 返回 (seg, cls)
        - seg 可能是 Tensor 或 list[Tensor] (deep supervision)
        这里统一转成: seg_logits: Tensor, cls_logits: Tensor
        """
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None

        # ---------- 第一次正向 ----------
        out = self.network(x)

        # 兼容: 可能是 (seg, cls) 或 只有 seg
        if isinstance(out, (list, tuple)) and len(out) == 2:
            seg_out, cls_out = out
        else:
            seg_out, cls_out = out, None

        # seg_out 可能是 deep supervision list，取最高分辨率那一张
        if isinstance(seg_out, (list, tuple)):
            seg_out = seg_out[0]

        prediction_seg = seg_out
        prediction_cls = cls_out
        num_predictions = 1

        # ---------- TTA 镜像 ----------
        if mirror_axes is not None:
            # x: [B, C, D, H, W]，镜像轴是空间轴，最多到 ndim-3
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'
            # nnU-Net 约定 mirror_axes 是不带通道的空间轴，从 0 开始，需要 +2
            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]

            for axes in axes_combinations:
                out_m = self.network(torch.flip(x, axes))

                if isinstance(out_m, (list, tuple)) and len(out_m) == 2:
                    seg_m, cls_m = out_m
                else:
                    seg_m, cls_m = out_m, None

                if isinstance(seg_m, (list, tuple)):
                    seg_m = seg_m[0]

                # 先网络 -> 再 flip 回来
                seg_m = torch.flip(seg_m, axes)
                prediction_seg += seg_m

                if cls_m is not None:
                    if prediction_cls is None:
                        prediction_cls = cls_m
                    else:
                        prediction_cls += cls_m

                num_predictions += 1

        # ---------- 平均 ----------
        prediction_seg = prediction_seg / num_predictions
        if prediction_cls is not None:
            prediction_cls = prediction_cls / num_predictions
        else:
            # 如果某些模型没 cls 分支，兜底造一个 0 向量，防止后面代码崩
            b = x.shape[0]
            num_cls = getattr(self.network, 'cls_num_classes', 1)
            prediction_cls = torch.zeros(b, num_cls, device=x.device, dtype=prediction_seg.dtype)

        return prediction_seg, prediction_cls



    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')
        # --- GAI: 为分类结果添加一个累加器与权重 ---
        all_cls_predictions = []
        cls_weights = []
        def producer(d, slh, q):
            for s in slh:
                q.put((torch.clone(d[s][None], memory_format=torch.contiguous_format).to(self.device), s))
            q.put('end')

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)
            queue = Queue(maxsize=2)
            t = Thread(target=producer, args=(data, slicers, queue))
            t.start()

            # preallocate arrays
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            # --- GAI: preallocate arrays (分割部分保持不变) ---
            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                           dtype=torch.half,
                                           device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)
            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')

            with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm) as pbar:
                while True:
                    item = queue.get()
                    if item == 'end':
                        queue.task_done()
                        break
                    workon, sl = item
                    # --- GAI: 从 TTA 获取 (seg, cls) ---
                    prediction_seg, prediction_cls = self._internal_maybe_mirror_and_predict(workon)
                    prediction_seg = prediction_seg.to(results_device)
                    
                    # 累加分类预测
                    all_cls_predictions.append(prediction_cls.to(results_device))
                    # 计算该 patch 的前景权重（用于 cls 聚合加权）：使用非背景概率的均值
                    try:
                        seg_logits_patch = prediction_seg  # [1, C, D, H, W]
                        probs_patch = torch.softmax(seg_logits_patch, dim=1)
                        # 非背景概率（1 - 背景通道）
                        fg_prob = 1.0 - probs_patch[:, 0:1]
                        w = float(fg_prob.mean().detach().cpu().item())
                    except Exception:
                        w = 1.0
                    cls_weights.append(max(w, 1e-6))
                    # --- GAI 结束 ---

                    if self.use_gaussian:
                        prediction_seg *= gaussian
                    
                    #predicted_logits [sl] is [3, 64, 128, 192], prediction_seg is [1, 3, 64, 128, 192]
                    predicted_logits[sl] += prediction_seg[0]
                    n_predictions[sl[1:]] += gaussian
                    queue.task_done()
                    pbar.update()
            queue.join()

            # predicted_logits /= n_predictions
            torch.div(predicted_logits, n_predictions, out=predicted_logits)
            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                   'predicted_logits to fp32')
            # --- GAI: 聚合分类结果（前景占比加权平均） ---
            # 堆叠所有补丁的预测 [Num_Patches, 1 (Batch_Size), Num_Classes]
            if all_cls_predictions:
                aggregated_cls_preds = torch.stack(all_cls_predictions)
                w = torch.tensor(cls_weights, device=results_device, dtype=aggregated_cls_preds.dtype)
                w = w / (w.sum() + 1e-8)
                final_cls_pred = (aggregated_cls_preds * w.view(-1, 1, 1)).sum(dim=0)
            else:
                # 理论上不应该发生，但作为回退
                final_cls_pred = torch.empty((0, 0), device=results_device) 
            # --- GAI 结束 ---
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits, final_cls_pred # GAI: 返回两个结果

    # --- GAI: 新增 - 基于分割ROI的分类TTA（仅分类，不计算分割重采样），软投票 ---
    @torch.inference_mode()
    def _classify_with_roi_tta(self,
                               data: torch.Tensor,
                               seg_logits: torch.Tensor,
                               use_extended_tta: bool = True) -> torch.Tensor:
        """
        输入：
          - data: 预处理后的图像张量 [C, D, H, W]
          - seg_logits: 分割logits [C_seg, D, H, W]（与 data 对齐的网络空间）
        过程：
          - 用 seg_logits 生成 ROI mask：roi = 1 - softmax(seg)[bg]
          - 滑窗遍历 patch，对每个 patch 进行分类前向。分类时传入 roi_patch 作为 ROI。
          - TTA：默认包含所有镜像组合；若 use_extended_tta=True，再额外加入 90° 旋转（D-H / H-W / D-W 3 个平面，各 90/270）。
          - 聚合：对每个 patch，先对 TTA 取平均得到 patch logits，再以 ROI 面积为权重做 patch 级加权平均。
        返回：病例级 cls logits [1, num_cls]
        """

        assert isinstance(data, torch.Tensor) and isinstance(seg_logits, torch.Tensor)
        device_compute = self.device if self.perform_everything_on_device else torch.device('cpu')

        # 构造 ROI mask（连续 [0,1]）
        with torch.autocast(self.device.type, enabled=(self.device.type == 'cuda')) if self.device.type == 'cuda' else dummy_context():
            seg_probs = torch.softmax(seg_logits.to(device_compute), dim=0)  # [C_seg, D, H, W]
            # 使用病灶通道作为 ROI 概率（默认取网络中的 lesion_channel_idx；若不存在则取最后一通道）
            try:
                lesion_idx = int(getattr(self.network, 'lesion_channel_idx', seg_probs.shape[0] - 1))
            except Exception:
                lesion_idx = seg_probs.shape[0] - 1
            lesion_idx = max(0, min(lesion_idx, seg_probs.shape[0] - 1))
            roi_full = seg_probs[lesion_idx:lesion_idx + 1].clamp(0, 1)  # [1, D, H, W]
            roi_full = roi_full.unsqueeze(0)  # [1,1,D,H,W]

        data = data.to(device_compute)
        slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

        # 延迟生成 TTA 变换列表：等拿到第一个 patch 再依 shape 决定
        tta_transforms = None
        pending_add_rotations = use_extended_tta

        # 分类累加器
        cls_sum = None
        w_sum = 0.0

        def apply_transform(x5d: torch.Tensor, r5d: torch.Tensor, t: dict):
            yx = x5d
            yr = r5d
            if "rot" in t:
                dims, k = t["rot"]
                yx = torch.rot90(yx, k=k, dims=dims)
                yr = torch.rot90(yr, k=k, dims=dims)
            if "flip" in t and len(t["flip"]) > 0:
                yx = torch.flip(yx, t["flip"]) 
                yr = torch.flip(yr, t["flip"]) 
            # 额外安全：将空间维度补齐到网络下采样总步长的倍数，避免奇偶导致的解码器对齐误差
            try:
                import numpy as _np
                strides = _np.array(self.configuration_manager.pool_op_kernel_sizes)  # [n_stages, dims]
                factors = _np.prod(strides, axis=0).astype(int).tolist()
            except Exception:
                factors = None
            if factors is not None:
                if yx.ndim == 5:
                    d, h, w = yx.shape[2], yx.shape[3], yx.shape[4]
                    fd, fh, fw = (factors + [1, 1, 1])[:3]
                    pd = (fd - (d % fd)) % fd
                    ph = (fh - (h % fh)) % fh
                    pw = (fw - (w % fw)) % fw
                    if pd or ph or pw:
                        yx = torch.nn.functional.pad(yx, (0, pw, 0, ph, 0, pd))
                        yr = torch.nn.functional.pad(yr, (0, pw, 0, ph, 0, pd))
                elif yx.ndim == 4:
                    h, w = yx.shape[2], yx.shape[3]
                    # 2D 配置只取后两维的因子
                    fh, fw = (factors[-2:]) if len(factors) >= 2 else (1, 1)
                    ph = (fh - (h % fh)) % fh
                    pw = (fw - (w % fw)) % fw
                    if ph or pw:
                        yx = torch.nn.functional.pad(yx, (0, pw, 0, ph))
                        yr = torch.nn.functional.pad(yr, (0, pw, 0, ph))
            return yx, yr

        # 生产者：patch 入队
        def producer_full(d, slh, q):
            for s in slh:
                q.put((torch.clone(d[s][None], memory_format=torch.contiguous_format).to(device_compute), s))
            q.put('end')

        queue = Queue(maxsize=2)
        t = Thread(target=producer_full, args=(data, slicers, queue))
        t.start()

        with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm) as pbar:
            while True:
                item = queue.get()
                if item == 'end':
                    queue.task_done()
                    break
                workon, sl = item  # workon: [1,C,Dp,Hp,Wp]
                spatial_sl = sl[1:]
                roi_patch = roi_full[(slice(None), slice(None), *spatial_sl)].to(device_compute)  # [1,1,Dp,Hp,Wp]

                # 对该 patch 做多次 TTA 分类，取平均
                patch_cls_sum = None
                success_cnt = 0
                # 懒构建 TTA 列表（包含镜像；可选添加平面内旋转）
                if tta_transforms is None:
                    mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
                    mirror_axes = [] if mirror_axes is None else list(mirror_axes)
                    # 将允许的 mirror 轴限制在当前 patch 的空间维度范围内
                    spatial_ndim = workon.ndim - 2  # 2D:2, 3D:3
                    m_valid = [a for a in mirror_axes if 0 <= a < spatial_ndim]
                    # map 到张量维度索引（通道之后）
                    m_axes = [a + 2 for a in m_valid]
                    combos = [()]
                    for i in range(len(m_axes)):
                        for c in itertools.combinations(m_axes, i + 1):
                            combos.append(c)
                    tta_transforms = [{"flip": c} for c in combos]

                    # 可选：添加平面内旋转（仅当 H==W 时更安全；避免不对称步长/奇偶导致的形状对齐问题）
                    if pending_add_rotations:
                        rot_dims = (3, 4) if workon.ndim == 5 else (2, 3)
                        for k in (1, 3):  # 90 和 270
                            tta_transforms.append({"flip": (), "rot": (rot_dims, k)})
                        pending_add_rotations = False

                for tform in tta_transforms:
                    x_t, r_t = apply_transform(workon, roi_patch, tform)
                    try:
                        try:
                            out = self.network(x_t, roi_mask=r_t)
                        except TypeError:
                            # 兼容不支持 roi_mask 的模型
                            out = self.network(x_t)
                        if isinstance(out, (list, tuple)) and len(out) == 2:
                            _, cls_logits = out
                        else:
                            continue
                        patch_cls_sum = cls_logits if patch_cls_sum is None else (patch_cls_sum + cls_logits)
                        success_cnt += 1
                    except Exception as _e:
                        # 某些旋转可能触发形状不匹配，忽略该 TTA 项
                        continue

                if patch_cls_sum is not None and success_cnt > 0:
                    patch_cls_mean = patch_cls_sum / float(success_cnt)
                    # 用 ROI 面积作为权重（避免全背景patch影响过大）
                    w = float(roi_patch.mean().detach().cpu().item())
                    w = max(w, 1e-6)
                    cls_sum = patch_cls_mean * w if cls_sum is None else (cls_sum + patch_cls_mean * w)
                    w_sum += w

                queue.task_done()
                pbar.update()
        queue.join()

        if cls_sum is None or w_sum <= 0:
            # 兜底：返回全零
            b = 1
            num_cls = getattr(self.network, 'cls_num_classes', 1)
            return torch.zeros((b, num_cls), device=device_compute)
        # 由于上面 patch_cls_sum 未除以 TTA 次数，这里将其缩放回平均：
        return cls_sum / w_sum

    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) \
            -> Union[np.ndarray, torch.Tensor]:
        assert isinstance(input_image, torch.Tensor)
        self.network = self.network.to(self.device)
        self.network.eval()

        empty_cache(self.device)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)
        # and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False
        # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

            if self.verbose:
                print(f'Input shape: {input_image.shape}')
                print("step_size:", self.tile_step_size)
                print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                       'constant', {'value': 0}, True,
                                                       None)

            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

            if self.perform_everything_on_device and self.device != 'cpu':
                try:
                    # --- GAI: 接收 (seg, cls) 元组 ---
                    predicted_logits, predicted_cls = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                       self.perform_everything_on_device)
                except RuntimeError:
                    print(
                        'Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')
                    empty_cache(self.device)
                    predicted_logits, predicted_cls = self._internal_predict_sliding_window_return_logits(data, slicers, False)
            else:
                predicted_logits, predicted_cls = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                   self.perform_everything_on_device)
            empty_cache(self.device)
            # revert padding
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
            
            # --- GAI: 返回两个结果 ---
            return predicted_logits, predicted_cls

    def predict_from_files_sequential(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           folder_with_segs_from_prev_stage: str = None):
        """
        Just like predict_from_files but doesn't use any multiprocessing. Slow, but sometimes necessary
        """
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
            if len(output_folder) == 0:  # just a file was given without a folder
                output_folder = os.path.curdir
        else:
            output_folder = None

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files_sequential).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(
                my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
            recursive_fix_for_json_export(my_init_kwargs)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

            # we need these two if we want to do things with the predictions like for example apply postprocessing
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        #######################

        # check if we need a prediction from the previous stage
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
                f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
                f' they are located via folder_with_segs_from_prev_stage'

        # sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, 0, 1,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        label_manager = self.plans_manager.get_label_manager(self.dataset_json)
        preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose)

        if output_filename_truncated is None:
            output_filename_truncated = [None] * len(list_of_lists_or_source_folder)
        if seg_from_prev_stage_files is None:
            seg_from_prev_stage_files = [None] * len(seg_from_prev_stage_files)

        ret = []
        for li, of, sps in zip(list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files):
            data, seg, data_properties = preprocessor.run_case(
                li,
                sps,
                self.plans_manager,
                self.configuration_manager,
                self.dataset_json
            )

            print(f'perform_everything_on_device: {self.perform_everything_on_device}')

            prediction = self.predict_logits_from_preprocessed_data(torch.from_numpy(data)).cpu()

            if of is not None:
                export_prediction_from_logits(prediction, data_properties, self.configuration_manager, self.plans_manager,
                  self.dataset_json, of, save_probabilities)
            else:
                ret.append(convert_predicted_logits_to_segmentation_with_correct_shape(prediction, self.plans_manager,
                     self.configuration_manager, self.label_manager,
                     data_properties,
                     save_probabilities))

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return ret

def _getDefaultValue(env: str, dtype: type, default: any,) -> any:
    try:
        val = dtype(os.environ.get(env) or default)
    except:
        val = default
    return val

def predict_entry_point_modelfolder():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-m', type=str, required=True,
                        help='Folder in which the trained model is. Must have subfolders fold_X for the different '
                             'folds you trained')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', '--c', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = nnUNetPredictor(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                allow_tqdm=not args.disable_progress_bar,
                                verbose_preprocessing=args.verbose)
    predictor.initialize_from_trained_model_folder(args.m, args.f, args.chk)

    # 记录推理开始时间（不包含模型加载）
    # start_time = time()

    predictor.predict_from_files(args.i, args.o, save_probabilities=args.save_probabilities,
                                 overwrite=not args.continue_prediction,
                                 num_processes_preprocessing=args.npp,
                                 num_processes_segmentation_export=args.nps,
                                 folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                 num_parts=1, part_id=0)

    # 打印总推理时间（包括预处理、网络前向与结果导出）
    total_time = time() - start_time
    # total_time = time() - start_time
    # print(f"Total inference time: {total_time:.2f} seconds")


def predict_entry_point():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-d', type=str, required=True,
                        help='Dataset with which you would like to predict. You can specify either dataset name or id')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. Specify the plans in which the desired configuration is located. '
                             'Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='What nnU-Net trainer class was used for training? Default: nnUNetTrainer')
    parser.add_argument('-c', type=str, required=True,
                        help='nnU-Net configuration that should be used for prediction. Config must be located '
                             'in the plans specified with -p')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=_getDefaultValue('nnUNet_npp', int, 3),
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=_getDefaultValue('nnUNet_nps', int, 3),
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-num_parts', type=int, required=False, default=1,
                        help='Number of separate nnUNetv2_predict call that you will be making. Default: 1 (= this one '
                             'call predicts everything)')
    parser.add_argument('-part_id', type=int, required=False, default=0,
                        help='If multiple nnUNetv2_predict exist, which one is this? IDs start with 0 can end with '
                             'num_parts - 1. So when you submit 5 nnUNetv2_predict calls you need to set -num_parts '
                             '5 and use -part_id 0, 1, 2, 3 and 4. Simple, right? Note: You are yourself responsible '
                             'to make these run on separate GPUs! Use CUDA_VISIBLE_DEVICES (google, yo!)')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    # slightly passive aggressive haha
    assert args.part_id < args.num_parts, 'Do you even read the documentation? See nnUNetv2_predict -h.'

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = nnUNetPredictor(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                verbose_preprocessing=args.verbose,
                                allow_tqdm=not args.disable_progress_bar)
    predictor.initialize_from_trained_model_folder(
        model_folder,
        args.f,
        checkpoint_name=args.chk
    )

    # 记录推理开始时间（不包含模型加载）
    start_time = time()

    run_sequential = args.nps == 0 and args.npp == 0
    
    if run_sequential:

        print("Running in non-multiprocessing mode")
        predictor.predict_from_files_sequential(args.i, args.o, save_probabilities=args.save_probabilities,
                                                overwrite=not args.continue_prediction,
                                                folder_with_segs_from_prev_stage=args.prev_stage_predictions)
    
    else:
        
        predictor.predict_from_files(args.i, args.o, save_probabilities=args.save_probabilities,
                                    overwrite=not args.continue_prediction,
                                    num_processes_preprocessing=args.npp,
                                    num_processes_segmentation_export=args.nps,
                                    folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                    num_parts=args.num_parts,
                                    part_id=args.part_id)

    # 打印总推理时间（包括预处理、网络前向与结果导出）
    total_time = time() - start_time
    total_time = time() - start_time
    print(f"Total inference time: {total_time:.2f} seconds")
    
    # r = predict_from_raw_data(args.i,
    #                           args.o,
    #                           model_folder,
    #                           args.f,
    #                           args.step_size,
    #                           use_gaussian=True,
    #                           use_mirroring=not args.disable_tta,
    #                           perform_everything_on_device=True,
    #                           verbose=args.verbose,
    #                           save_probabilities=args.save_probabilities,
    #                           overwrite=not args.continue_prediction,
    #                           checkpoint_name=args.chk,
    #                           num_processes_preprocessing=args.npp,
    #                           num_processes_segmentation_export=args.nps,
    #                           folder_with_segs_from_prev_stage=args.prev_stage_predictions,
    #                           num_parts=args.num_parts,
    #                           part_id=args.part_id,
    #                           device=device)


if __name__ == '__main__':
    ########################## predict a bunch of files
    from nnunetv2.paths import nnUNet_results, nnUNet_raw

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset004_Hippocampus/nnUNetTrainer_5epochs__nnUNetPlans__3d_fullres'),
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )
    # predictor.predict_from_files(join(nnUNet_raw, 'Dataset003_Liver/imagesTs'),
    #                              join(nnUNet_raw, 'Dataset003_Liver/imagesTs_predlowres'),
    #                              save_probabilities=False, overwrite=False,
    #                              num_processes_preprocessing=2, num_processes_segmentation_export=2,
    #                              folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
    #
    # # predict a numpy array
    # from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    #
    # img, props = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTr/liver_63_0000.nii.gz')])
    # ret = predictor.predict_single_npy_array(img, props, None, None, False)
    #
    # iterator = predictor.get_data_iterator_from_raw_npy_data([img], None, [props], None, 1)
    # ret = predictor.predict_from_data_iterator(iterator, False, 1)

    ret = predictor.predict_from_files_sequential(
        [['/media/isensee/raw_data/nnUNet_raw/Dataset004_Hippocampus/imagesTs/hippocampus_002_0000.nii.gz'], ['/media/isensee/raw_data/nnUNet_raw/Dataset004_Hippocampus/imagesTs/hippocampus_005_0000.nii.gz']],
        '/home/isensee/temp/tmp', False, True, None
    )


