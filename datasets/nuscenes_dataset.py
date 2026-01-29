# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import os.path as osp
from mmcv import print_log
import json


@DATASETS.register_module()
class HFLNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, 
                 *args, 
                 return_gt_info=False, 
                 scenes=None, 
                 scene_translation=None,
                 enable_hfl=True, 
                 **kwargs):
        super(HFLNuScenesDataset, self).__init__(*args, **kwargs)
        self.return_gt_info = return_gt_info

        if scenes is None:
            raise ValueError(f"No scenes specified.")
        if scene_translation is None:
            raise ValueError(f"No file provided to translate scene tokens to scene names")

        with open(scene_translation, "r") as f:
            token_to_name = json.load(f)

        # Convert to set for faster look-up
        scenes = set(scenes)
        org_len = len(self.data_infos)
        self.data_infos = [
            info for info in self.data_infos
            if token_to_name.get(info["scene_token"]) in scenes
        ]

        if self.filter_empty_gt:
            def has_gt(info):
                gt_names = info.get('gt_names', None)
                return gt_names is not None and len(gt_names) > 0

            before = len(self.data_infos)
            self.data_infos = [info for info in self.data_infos if has_gt(info)]
            after = len(self.data_infos)

            print_log(
                f">>> Filtered empty-GT samples: {before} -> {after}",
                logger='root'
            )

        if len(self.data_infos) == 0:
            raise ValueError(f"No samples are available for "
                                f"specified scenes")

        self.flag = np.zeros(len(self.data_infos), dtype=np.uint8)

        print_log(f">>> dataset length: {len(self.data_infos)}", logger='root')


    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]

        lidar_path = info['lidar_path']
        if not osp.isabs(lidar_path):
            lidar_path = osp.join(self.data_root, lidar_path)

        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            # pts_filename=info['lidar_path'],
            pts_filename=lidar_path,
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
            img_sweeps=None if 'img_sweeps' not in info else info['img_sweeps'],
            radar_info=None if 'radars' not in info else info['radars']
        )

        if self.return_gt_info:
            input_dict['info'] = info

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            img_timestamp = []
            for cam_type, cam_info in info['cams'].items():
                img_timestamp.append(cam_info['timestamp'] / 1e6)
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

