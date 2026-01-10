from hfl.coordinator import Coordinator

from mmcv import print_log


def main():
    print_log(f"Constructing coordinator...", logger='root' )

    cloud = Coordinator(
        work_root = "/tudelft.net/staff-umbrella/rdramautar/HFL/experiments/exp_001",
        base_cfg_path = "/tudelft.net/staff-umbrella/rdramautar/HFL/configs/cmt_lidar_voxel0075_cbgs.py",
        manifest_path = "/tudelft.net/staff-umbrella/rdramautar/HFL/data/dataset_distribution.json",
        init_ckpt_path = "/tudelft.net/staff-umbrella/rdramautar/CMT-BTSA/experiments/experiments_5/epoch_10.pth",
        num_local_rounds = 1,
        num_edge_rounds = 1,
        num_global_rounds = 1,
        token_to_name_path = "/tudelft.net/staff-umbrella/rdramautar/HFL/data/scene_name_to_token.json",
        seed = 0
    )
    print_log(f"Coordinator constructed. Starting training...", logger='root' )

    cloud.train()

if __name__ == '__main__':
    main()