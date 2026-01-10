from pathlib import Path
from mmcv import Config
import json

test_idx = 2

WORK_ROOT = f"/tudelft.net/staff-umbrella/rdramautar/CMT-BTSA/workdirs/federated_test_{test_idx}/"
# Path to base config file
BASE_CONFIG = "/tudelft.net/staff-umbrella/rdramautar/CMT-BTSA/configs/federated/cmt_lidar_voxel0075_cbgs.py"

# Index of global training round
GLOBAL_ROUND = 0

def build_client_cfg(base_cfg, )

def main():
    #  Build base config used by all clients
    base_cfg = Config.fromfile()

    # Load Dict that assigns scenes to clients
    with open(DATA_ASSIGNMENT, "r") as f:
        data_assignment = json.load(f)
    # List of keys describing edges
    edges = data_assignment["edges"]

    # Edge aggregation round
    for edge_name, edge_data in edges.items():
        # Get clients assigned to edge server
        clients = edge_data["clients"]
        
        # Sequential client training
        for client_id, client in clients.items():
            # nuScenes scenes assigned to client
            scenes = client["scenes"]

            # Path where client's training results will be stored
            client_workdir = Path(WORK_ROOT) / f"{edge_name}" \
                                             / f"client_{client_id}" \
                                             / f"round_{GLOBAL_ROUND}"
            client_work_dir.mkdir(parents=True, exists_ok=True)


            





if __name__=="__main__":
    main()