from hfl.aggregation import save_state_dict, average_weights
from hfl.client import Client
from hfl.lr import LRScheduler

import torch

from typing import Optional, Union
from pathlib import Path
import gc

from mmcv import print_log, Config


class Edge():
    """ Edge server.
    
    Args:
        name (str): Name or location of edge server.
        clients(Dict): Mapping of clients to nuScenes scenes.
        work_root (str | Path): Root path used in config, but not necessarily the location
            where training results are stored.
        base_cfg_path (str): File path to model config template.
        num_rounds (int): Number of edge training iterations before global aggregation.
            Default: 1
        num_local_rounds (int): Number of training epochs performed by clients.
            Default: 1
        token_to_name_path (Optional[str]): File path to json storing mapping from
            nuScenes scene name and scene token.
        seed (int): Training seed for deterministic training.
            Default: 0
    """
    def __init__(self,
                 name: str,
                 clients: dict,
                 base_cfg: Config,
                 lr_cfg: dict,
                 num_rounds: int = 1,
                 num_local_rounds: int = 1,
                 token_to_name_path: Optional[str] = None,
                 seed: int = 0):

        if num_rounds <= 0: 
            raise ValueError("num_rounds must be a positive integer")
        if num_local_rounds <= 0: 
            raise ValueError("num_local_rounds must be a positive integer")

        self.name = name
        self.base_cfg = base_cfg
        self.num_rounds = num_rounds

        self.round = 0
        self.lr_scheduler = LRScheduler(**lr_cfg)

        # Create clients
        self.clients = []
        for client_name, client_info in clients.items():
            print_log(f"Instantiating client {client_name}", logger='root')
            client = Client(
                name = client_name,
                scenes = client_info['scenes'],
                num_epochs = num_local_rounds,
                token_to_name_path = token_to_name_path,
                seed = seed
            )
            self.clients.append(client)


    def _single_iter(self, load_path: Union[str, Path], edge_root: Path):
        weight_paths = []
        sample_counts = []

        lr = self.lr_scheduler.lr_at(self.round)
        print_log(f"Using learning rate: {lr}", logger='root' )

        for client in self.clients[:3]:
            client_root = edge_root / str(client.name)
            client_root.mkdir(parents=True, exist_ok=True)

            try:
                save_path, num_samples = client.train(self.base_cfg, load_path, client_root, lr)
            finally:
                # Free Python objects and cached GPU memory in between clients
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(f"    - {client.name} finished training")

            weight_paths.append(save_path)
            sample_counts.append(num_samples)

        self.round += 1

        return weight_paths, sample_counts



    def train(self, 
              load_path: Union[str, Path], 
              work_root: Union[str, Path]):
        """Train client models and aggregated updated client weights

        Args:
            load_path (str | Path): File path of the initial weights.
            work_root (str | Path): Folder path where the edge aggregation results will
                be stored.
        """
        total_samples = 0
        for i in range(self.num_rounds):
            print_log(f"[[EDGE - {self.name}] - Round {i}", logger='root' )
            # Create folder for i-th edge training round
            edge_root = Path(work_root) / f"edge_round_{i}"
            edge_path = edge_root / "edge_weights.pth"
            edge_root.mkdir(parents=True, exist_ok=True)

            # Train across all clients
            weight_paths, sample_counts = self._single_iter(load_path, edge_root)
            
            # Aggregate client weights
            avg_weights = average_weights(weight_paths, sample_counts)
            total_samples += sum(sample_counts)

            # Store aggregated weights
            save_state_dict(avg_weights, edge_path)
            load_path = edge_path

        return edge_path, total_samples


