from hfl.aggregation import average_weights, save_state_dict
from hfl.edge import Edge

from typing import Optional, Union
from pathlib import Path
import json

from mmcv import print_log
from mmcv import Config


class Coordinator():
    """ Cloud server coordinating the federated training.

    Args:
        work_root (str): Folder path where all training results will be stored.
        base_cfg_path (str): File path to model config template.
        manifest_path (str): File path to json assigning clients to edges and data samples
            to clients.
        init_ckpth_path (str): File path (.pth) to initial weights.
        num_local_rounds (int): Number of client training rounds (epochs).
            Default: 1
        num_edge_rounds (int): Number of edge training rounds before global aggregation.
            Default: 1
        num_global_rounds (int): Number of global training rounds.
            Default: 1
        token_to_name_path (Optional[str]): File path to json file which maps scene token
            to scene name. Optional, as it can be included in the config file.
        seed (int): Training seed for deterministic training.
            Default: 0
    """
    def __init__(self,
                work_root: str,
                base_cfg_path: str,
                manifest_path: str,
                init_ckpt_path: str,
                lr_cfg: dict,
                num_local_rounds: int = 1,
                num_edge_rounds: int = 1,
                num_global_rounds: int = 1,
                token_to_name_path: Optional[str] = None,
                seed: int = 0):

        if num_local_rounds <= 0: 
            raise ValueError("num_local_rounds must be a positive integer")
        if num_edge_rounds <= 0: 
            raise ValueError("num_edge_rounds must be a positive integer")
        if num_global_rounds <= 0: 
            raise ValueError("num_global_rounds must be a positive integer")

        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)

        self.init_ckpt_path = init_ckpt_path
        self.num_global_rounds = num_global_rounds

        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)
        
        print_log(f"Read manifest", logger='root' )

        # List of keys describing edges
        edges = self.manifest["edges"]

        base_cfg = Config.fromfile(base_cfg_path)
        print_log(f"Created base config file", logger='root' )

        # Create edge servers
        self.edges = []
        for edge_name, edge_data in edges.items():
            print_log(f"Instantiating Edge server {edge_name}", logger='root' )
            edge = Edge(
                name = edge_name,
                clients = edge_data["clients"],
                base_cfg = base_cfg,
                num_rounds = num_edge_rounds,
                num_local_rounds = num_local_rounds,
                token_to_name_path = token_to_name_path,
                seed = seed,
                lr_cfg = lr_cfg
            )
            self.edges.append(edge)


    def _single_iter(self, load_path: Union[str, Path], global_root):
        weight_paths = []
        sample_counts = []
        for edge in self.edges:
            edge_root = global_root / str(edge.name)
            edge_root.mkdir(parents=True, exist_ok=True)

            save_path, num_samples = edge.train(load_path, edge_root)

            weight_paths.append(save_path)
            sample_counts.append(num_samples)

        return weight_paths, sample_counts


    def train(self):
        load_path = self.init_ckpt_path
        for i in range(self.num_global_rounds):
            print_log(f"[CLOUD] - Round {i}", logger='root' )
            global_root = self.work_root / f"global_round_{i}"
            global_path = global_root / "global_weights.pth"
            global_root.mkdir(parents=True, exist_ok=True)

            # Train across all edges
            weight_paths, sample_counts = self._single_iter(load_path, global_root)

            # Aggregate edge weights
            avg_weights = average_weights(weight_paths, sample_counts)

            # Store aggregated weights
            save_state_dict(avg_weights, global_path)
            load_path = global_path
