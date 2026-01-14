from mmdet3d_plugin.datasets.custom_nuscenes_dataset import CustomNuScenesBTSA

import numpy as np
from mmdet.datasets import DATASETS
from mmcv.parallel import DataContainer as DC
import torch
import json
from mmcv import print_log
import os.path as osp

@DATASETS.register_module()
class HFLNuScenesBTSA(CustomNuScenesBTSA):
    def __init__(self, *args, scenes=None, scene_translation=None, 
                    enable_hfl=False, **kwargs):
        """ nuScenes dataloader for use in FL.

        Args:
            scenes (List[str]): human-readable scene names assigned to a specific client
        """
        super().__init__(*args, **kwargs)

        if not enable_hfl:
            return

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

    def prepare_train_data(self, curr_idx):
        """ Generate a training data sample

        Args:
            curr_idx (int): Index of the sampled frame from the entire training set.

        Returns:
            Dict(type=Collect3D) containing frame information of all frames in temporal \
                queue
        """
        # Identify relative index of center frame in temporal queue
        half = self.queue_length // 2

        # Store indices of all frames in temporal queue relative to 
        # entire training set
        indices = [i for i in range(curr_idx - half, curr_idx + half + 1)]

        # Get info of sampled frame
        curr_info = self.data_infos[curr_idx]
        assert curr_info is not None
                
        queue = []
        # for i, idx in enumerate(indices):
        for idx in indices:
            # Check if index exists in training set
            if (idx < 0) or (idx >= len(self.data_infos)):
                queue.append(None)
                continue

            # Check if supporting frame belongs to the same scene as the 
            # sampled frame
            same_scene = bool(curr_info['scene_token'] == 
                              self.data_infos[idx]['scene_token'])
            if not same_scene:
                queue.append(None)
                continue

            sample_info = self.get_img_metas(idx)

            # Ensure img_metas of sampled frame satisfies criteria
            if idx == curr_idx:
                if sample_info is None:
                    raise ValueError(f"No metas could be found for sampled index")

                if self.filter_empty_gt:
                    labels = sample_info['gt_labels_3d']._data
                    if not (labels != -1).any():
                        raise ValueError(f"No labels are available for sampled index")

            if sample_info is None:
                queue.append(None)
            else:
                queue.append(sample_info)

        return self.union2one(queue, half)
