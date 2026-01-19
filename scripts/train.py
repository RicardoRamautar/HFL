from hfl.coordinator import Coordinator

from mmcv import print_log


def main():
    print_log(f"Constructing coordinator...", logger='root' )

    cloud = Coordinator(
        work_root = "/tudelft.net/staff-umbrella/rdramautar/HFL/experiments/exp_004",
        base_cfg_path = "/tudelft.net/staff-umbrella/rdramautar/HFL/configs/cmt_lidar_voxel0075_cbgs.py",
        manifest_path = "/tudelft.net/staff-umbrella/rdramautar/HFL/data/dataset_distribution.json",
        init_ckpt_path = "/tudelft.net/staff-umbrella/rdramautar/CMT-BTSA/experiments/experiments_7/best_pts_bbox_NuScenes/mAP_epoch_2.pth",
        lr_cfg = {
            'policy': "exp",
            'base_lr': 1e-4,
            'gamma': 0.95
        },
        num_local_rounds = 2,
        num_edge_rounds = 2,
        num_global_rounds = 2,
        token_to_name_path = "/tudelft.net/staff-umbrella/rdramautar/HFL/data/scene_name_to_token.json",
        seed = 0
    )
    print_log(f"Coordinator constructed. Starting training...", logger='root' )

    cloud.train()

if __name__ == '__main__':
    main()