import os
import warnings
from typing import Union, Tuple, List

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from threadpoolctl import threadpool_limits

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd


class nnUNetDataLoader(DataLoader):
    def __init__(self,
                 data: nnUNetBaseDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 task_mode: str,  
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...]] = None,
                 probabilistic_oversampling: bool = False,
                 transforms=None,
                 cls_patch_size: Union[List[int], Tuple[int, ...], np.ndarray, None] = None
                 ):
        """
        If we get a 2D patch size, make it pseudo 3D and remember to remove the singleton dimension before
        returning the batch
        """
        super().__init__(data, batch_size, 1, None, True,
                         False, True, sampling_probabilities)

        assert task_mode in ('seg_only', 'cls_only'), f"task_mode must be 'seg_only' or 'cls_only', got {task_mode}"
        self.task_mode = task_mode

        # 2D → 3D 伪 3D 处理
        if len(patch_size) == 2:
            final_patch_size = (1, *patch_size)
            patch_size = (1, *patch_size)
            self.patch_size_was_2d = True
        else:
            self.patch_size_was_2d = False

        # 这个 indices 是 DataLoader 抽样 case 用的
        self.indices = data.identifiers

        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = np.array(patch_size).astype(int)
        cls_patch_size = np.array([96, 160, 224], dtype=int)
        # cls 模式下使用单独的 patch size
        if self.task_mode == 'cls_only':
            if cls_patch_size is None:
                raise ValueError(
                    "task_mode='cls_only' 时必须显式传入 cls_patch_size=(D, H, W)，"
                    "你可以用前面的统计脚本拿到最大 ROI 尺寸后在这里硬编码。"
                )
            cls_patch_size = np.array(cls_patch_size).astype(int)
            if self.patch_size_was_2d:
                cls_patch_size = np.concatenate([[1], cls_patch_size])
            self.cls_patch_size = cls_patch_size
        else:
            self.cls_patch_size = None

        # need_to_pad 还是照原来的算（主要 seg 用）
        self.need_to_pad = (self.patch_size - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if self.patch_size_was_2d:
                pad_sides = (0, *pad_sides)
            for d in range(len(self.need_to_pad)):
                self.need_to_pad[d] += pad_sides[d]

        self.num_channels = None
        self.pad_sides = pad_sides
        self.sampling_probabilities = sampling_probabilities
        self.annotated_classes_key = tuple([-1] + label_manager.all_labels)
        self.has_ignore = label_manager.has_ignore_label
        self.get_do_oversample = self._oversample_last_XX_percent if not probabilistic_oversampling \
            else self._probabilistic_oversampling
        self.transforms = transforms

        # 最后再算 data_shape / seg_shape（cls 用 cls_patch_size）
        self.data_shape, self.seg_shape = self.determine_shapes(label_manager)

    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def _probabilistic_oversampling(self, sample_idx: int) -> bool:
        return np.random.uniform() < self.oversample_foreground_percent

    def determine_shapes(self, label_manager: LabelManager):
        # load one case 只是为了拿通道数
        data, seg, seg_prev, properties = self._data.load_case(self._data.identifiers[0])
        num_color_channels = data.shape[0]

        if self.task_mode == 'cls_only' and self.cls_patch_size is not None:
            patch = self.cls_patch_size
        else:
            patch = self.patch_size

        data_shape = (self.batch_size, num_color_channels, *patch)
        channels_seg = seg.shape[0]
        if seg_prev is not None:
            channels_seg += 1
        seg_shape = (self.batch_size, channels_seg, *patch)
        return data_shape, seg_shape

    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
        """
        只给 seg 模式用的随机 bbox（原版 nnUNet 逻辑）
        """
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        if not force_fg and not self.has_ignore:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
        else:
            if not force_fg and self.has_ignore:
                selected_class = self.annotated_classes_key
                if len(class_locations[selected_class]) == 0:
                    warnings.warn('Warning! No annotated pixels in image!')
                    selected_class = None
            elif force_fg:
                assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
                if overwrite_class is not None:
                    assert overwrite_class in class_locations.keys(), 'desired class ("overwrite_class") does not ' \
                                                                      'have class_locations (missing key)'
                eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                if any(tmp):
                    if len(eligible_classes_or_regions) > 1:
                        eligible_classes_or_regions.pop(np.where(tmp)[0][0])

                if len(eligible_classes_or_regions) == 0:
                    selected_class = None
                    if verbose:
                        print('case does not contain any foreground classes')
                else:
                    selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                        (overwrite_class is None or (overwrite_class not in eligible_classes_or_regions)) else overwrite_class
            else:
                raise RuntimeError('lol what!?')

            if selected_class is not None:
                voxels_of_that_class = class_locations[selected_class]
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
        return bbox_lbs, bbox_ubs

    def generate_train_batch(self):
        selected_keys = self.get_indices()

        # =========================
        # CLS 模式：固定 cls_patch_size，中心对齐 + pad，不随机 crop
        # =========================
        if self.task_mode == 'cls_only':
            data_all = np.zeros(self.data_shape, dtype=np.float32)
            class_labels_all = np.zeros(self.batch_size, dtype=np.int64)

            for j, i in enumerate(selected_keys):
                # 解析分类标签 quiz_LABEL_xxx
                try:
                    class_label = int(i.split('_')[1])
                    class_labels_all[j] = class_label
                except (IndexError, ValueError):
                    raise RuntimeError(
                        f"Failed to parse class label from case identifier: '{i}'. "
                        f"Expected format 'quiz_LABEL_CASEID'."
                    )

                data, seg, seg_prev, properties = self._data.load_case(i)
                img_shape = np.array(data.shape[1:])           # (D,H,W)
                patch = self.cls_patch_size                    # (pD,pH,pW)


                # 中心对齐的 bbox（可能会越界 → crop_and_pad_nd 自动 pad）
                bbox_lbs = []
                bbox_ubs = []
                for d in range(len(img_shape)):
                    lb = (img_shape[d] - patch[d]) // 2   # 可能是负数
                    ub = lb + patch[d]
                    bbox_lbs.append(int(lb))
                    bbox_ubs.append(int(ub))
                bbox = [[lb, ub] for lb, ub in zip(bbox_lbs, bbox_ubs)]

                data_all[j] = crop_and_pad_nd(data, bbox, 0)

            if self.patch_size_was_2d:
                data_all = data_all[:, :, 0]

            # transforms 只需要 image（seg 不用）
            if self.transforms is not None:
                with torch.no_grad():
                    with threadpool_limits(limits=1, user_api=None):
                        data_all = torch.from_numpy(data_all).float()
                        class_labels_all = torch.from_numpy(class_labels_all).long()

                        images = []
                        for b in range(self.batch_size):
                            tmp = self.transforms(**{'image': data_all[b]})
                            images.append(tmp['image'])
                        data_all = torch.stack(images)
                        del images
            else:
                data_all = torch.from_numpy(data_all).float()
                class_labels_all = torch.from_numpy(class_labels_all).long()

            return {
                'data': data_all,              # [B, C, *cls_patch_size]
                'class_label': class_labels_all,
                'target': [],
                'keys': selected_keys
            }

        # =========================
        # SEG 模式：原始 nnUNet + 你的 class_label
        # =========================
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        class_labels_all = np.zeros(self.batch_size, dtype=np.int64)

        for j, i in enumerate(selected_keys):
            try:
                class_label = int(i.split('_')[1])
                class_labels_all[j] = class_label
            except (IndexError, ValueError):
                class_labels_all[j] = -1

            force_fg = self.get_do_oversample(j)
            data, seg, seg_prev, properties = self._data.load_case(i)
            shape = data.shape[1:]

            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i_, j_] for i_, j_ in zip(bbox_lbs, bbox_ubs)]

            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)[None]))
            seg_all[j] = seg_cropped

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    class_labels_all = torch.from_numpy(class_labels_all).long()

                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images
        else:
            data_all = torch.from_numpy(data_all).float()
            seg_all = torch.from_numpy(seg_all).to(torch.int16)
            class_labels_all = torch.from_numpy(class_labels_all).long()

        return {
            'data': data_all,
            'target': seg_all,
            'class_label': class_labels_all,
            'keys': selected_keys
        }


if __name__ == '__main__':
    folder = join(nnUNet_preprocessed, 'Dataset002_Quiz', 'nnUNetPlans_3d_fullres')
    ds = nnUNetDatasetBlosc2(folder)
    pm = PlansManager(join(folder, os.pardir, 'nnUNetResEncUNetMPlans.json'))
    lm = pm.get_label_manager(load_json(join(folder, os.pardir, 'dataset.json')))

    # 假设你通过上面的统计拿到 max_shape = [96, 160, 224]
    cls_patch_size = (96, 160, 224)
    # p90: [ 78 146 214]
    # seg dataloader 测试
    dl_seg = nnUNetDataLoader(ds, 2, (16, 16, 16), (16, 16, 16), lm,
                              0.33, None, None, task_mode='seg')
    b_seg = next(dl_seg)
    print("seg batch keys:", b_seg.keys())
    print("seg data shape:", b_seg['data'].shape)

    # cls dataloader 测试（用 cls_patch_size）
    dl_cls = nnUNetDataLoader(ds, 2, (1, 1, 1), (1, 1, 1), lm,
                              0.0, None, None, task_mode='cls',
                              cls_patch_size=cls_patch_size)
    b_cls = next(dl_cls)
    print("cls batch keys:", b_cls.keys())
    print("cls data shape:", b_cls['data'].shape)
