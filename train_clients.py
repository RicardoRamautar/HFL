# import subprocess
# import yaml, json
# from mmcv import Config
# from pathlib import Path

# BASE_CONFIG = "/tudelft.net/staff-umbrella/rdramautar/CMT-BTSA/configs/federated/cmt_lidar_voxel0075_cbgs.py"
# TRAIN_SCRIPT = "/tudelft.net/staff-umbrella/rdramautar/CMT/tools/train.py"

# def run_client(client_id, scenes, work_dir, load_from):
#     cfg = Config.fromfile(BASE_CONFIG)

#     # Inject client-specific dataset config
#     cfg.data.train.enable_hfl = True
#     cfg.data.train.scenes = scenes
#     cfg.data.train.scene_translation = "/tudelft.net/staff-umbrella/rdramautar/HFMTL/scene_name_to_token.json"

#     # Set checkpoint init
#     cfg.load_from = load_from

#     # Write temp config
#     cfg_path = Path(work_dir) / "config.py"
#     cfg.dump(cfg_path)

#     # Call training script
#     subprocess.run([
#         "python", TRAIN_SCRIPT,
#         str(cfg_path),
#         "--work-dir", work_dir
#     ], check=True)

# def main():
#     with open('/tudelft.net/staff-umbrella/rdramautar/HFMTL/dataset_distribution.json') as f:
#         data_distribution = json.load(f)

#     edges = data_distribution.["edges"]

#     clients = data_distribution[edges[0]]
#     for client_id, client in clients.items():
#         run_client(
#             client_id=client_id,
#             scenes=client["scenes"],
#             work_dir=f"/tudelft.net/staff-umbrella/rdramautar/CMT-BTSA/workdirs/federated_test_1/client_{client_id}",
#             load_from=global_checkpoint
#         )

# if __name__ == "__main__":
#     main()

import subprocess
import json
from mmcv import Config
from pathlib import Path

BASE_CONFIG = "/tudelft.net/staff-umbrella/rdramautar/CMT-BTSA/configs/federated/cmt_lidar_voxel0075_cbgs.py"
TRAIN_SCRIPT = "/tudelft.net/staff-umbrella/rdramautar/CMT/tools/train.py"
SCENE_TRANSLATION = "/tudelft.net/staff-umbrella/rdramautar/HFMTL/scene_name_to_token.json"

# Round-0 global initialization
GLOBAL_CHECKPOINT = None  # or path to pretrained CMT

def run_client(client_id, scenes, work_dir, load_from):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(BASE_CONFIG)

    # Inject client-specific dataset config
    cfg.data.train.enable_hfl = True
    cfg.data.train.scenes = scenes
    cfg.data.train.scene_translation = SCENE_TRANSLATION

    # Initial weights
    if load_from is not None:
        cfg.load_from = load_from

    # Write temp config
    cfg_path = work_dir / "config.py"
    # cfg.dump(cfg_path)
    cfg.dump(str(cfg_path))

    # Launch training
    subprocess.run(
        [
            "python",
            TRAIN_SCRIPT,
            str(cfg_path),
            "--work-dir",
            str(work_dir),
        ],
        check=True,
    )

def main():
    with open("dataset_distribution.json") as f:
        manifest = json.load(f)

    edges = manifest["edges"]

    # For now: single edge, sequential clients
    for edge_name, edge_data in edges.items():
        clients = edge_data["clients"]

        for client_id, client in clients.items():
            print(f"=== Training {edge_name} / {client_id} ===")

            run_client(
                client_id=client_id,
                scenes=client["scenes"],
                work_dir=f"/tudelft.net/staff-umbrella/rdramautar/CMT-BTSA/workdirs/federated_test_2/client_{client_id}",
                load_from=GLOBAL_CHECKPOINT,
            )


if __name__ == "__main__":
    main()
