from hfl.coordinator import Coordinator
from mmcv import print_log

def main():
    print_log(f"Constructing coordinator...", logger='root' )

    base_lr = 0.0001 / 4
    num_local_rounds = 1
    num_edge_rounds = 2
    # num_global_rounds = 12
    num_global_rounds = 15

    cloud = Coordinator(
        work_root = "/tudelft.net/staff-umbrella/rdramautar/HFL/experiments/exp_031",
        # base_cfg_path = "/tudelft.net/staff-umbrella/rdramautar/HFL/configs/cmt_lidar_epoch_lr.py",
        # base_cfg_path = "/tudelft.net/staff-umbrella/rdramautar/HFL/configs/standard_cmt.py",
        base_cfg_path = "/tudelft.net/staff-umbrella/rdramautar/HFL/configs/cmt_lidar_cyclic_lr.py",
        # val_cfg_path = "/tudelft.net/staff-umbrella/rdramautar/HFL/configs/cmt_lidar_cyclic_lr_val.py",
        val_cfg_path = "/tudelft.net/staff-umbrella/rdramautar/HFL/configs/cmt_lidar_cyclic_lr.py",
        # manifest_path = "/tudelft.net/staff-umbrella/rdramautar/HFL/data/dataset_distribution.json",
        manifest_path = "/tudelft.net/staff-umbrella/rdramautar/HFL/data/iid_day_night_2edges_4clients.json",
        # manifest_path = "/tudelft.net/staff-umbrella/rdramautar/HFL/data/non_iid_distribution.json",
        init_ckpt_path = None,
        num_local_rounds = num_local_rounds,
        num_edge_rounds = num_edge_rounds,
        num_global_rounds = num_global_rounds,
        # lr_cfg = {
        #     'policy': "exp",
        #     'total_epochs': num_local_rounds*num_edge_rounds*num_global_rounds,
        #     'base_lr': 1e-4,
        #     'gamma': 0.95
        # },
        lr_cfg = {
            'policy': "cyclic",
            'total_epochs': num_local_rounds * num_edge_rounds * num_global_rounds,
            'initial_lr': base_lr,
            # 'min_lr': 0.0001*base_lr,
            # 'min_lr': 0.005*base_lr,
            'min_lr': 0.001*base_lr,
            'max_lr': 6*base_lr
        },
        token_to_name_path = "/tudelft.net/staff-umbrella/rdramautar/HFL/data/scene_name_to_token.json",
        seed = 0,
        # resume_from = 0        # Latest fully complete global round
    )
    print_log(f"Coordinator constructed. Starting training...", logger='root' )

    cloud.train()

if __name__ == '__main__':
    main()
