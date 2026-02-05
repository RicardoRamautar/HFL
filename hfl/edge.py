from hfl.aggregation import save_state_dict, average_weights
from hfl.client import Client
from hfl.lr import LRScheduler

import torch

from typing import Optional, Union
from pathlib import Path
import gc

from mmcv import print_log, Config
# import multiprocessing as mp

# from itertools import islice
# import os

# def _run_client(client, base_cfg, load_path, client_root, gpu_id, rounds):
#     # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

#     try:
#         return client.train(base_cfg, load_path, client_root, gpu_id, rounds)
#     finally:
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

# # def _wrapped_worker(client, base_cfg, load_path, edge_root_str, gpu_id, rounds, results):
# #     client_root = Path(edge_root_str) / client.name
# #     client_root.mkdir(parents=True, exist_ok=True)

# #     save_path, num_samples = _run_client(
# #         client, base_cfg, load_path, client_root, gpu_id, rounds
# #     )
# #     results.append((save_path, num_samples))

# def _wrapped_worker(client, base_cfg, load_path, edge_root_str, gpu_id, rounds, result_queue):
#     client_root = Path(edge_root_str) / client.name
#     client_root.mkdir(parents=True, exist_ok=True)

#     save_path, num_samples = _run_client(
#         client, base_cfg, load_path, client_root, gpu_id, rounds
#     )
#     # Send result back to parent (safe IPC)
#     result_queue.put((client.name, str(save_path), int(num_samples)))


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
        self.num_local_rounds = num_local_rounds

        self.round = 0
        # self.lr_scheduler = LRScheduler(**lr_cfg)

        # Create clients
        self.clients = []
        # self.client_rounds = {}
        for client_name, client_info in clients.items():
            print_log(f"Instantiating client {client_name}", logger='root')
            client = Client(
                name = client_name,
                scenes = client_info['scenes'],
                lr_cfg = lr_cfg,
                num_epochs = self.num_local_rounds,
                token_to_name_path = token_to_name_path,
                seed = seed
            )
            self.clients.append(client)

            # self.client_rounds[client_name] = 0


    def _single_iter(self, load_path: Union[str, Path], edge_root: Path):
        weight_paths = []
        sample_counts = []

        # lr = self.lr_scheduler.lr_at(self.round)
        # lr = {i: self.lr_scheduler.lr_at(i) for i in range(self.round, self.round+self.num_local_rounds)}
        # print_log(f"Using learning rate: {lr}", logger='root' )

        for client in self.clients[:2]:
            client_root = edge_root / str(client.name)
            client_root.mkdir(parents=True, exist_ok=True)

            try:
                # save_path, num_samples = client.train(self.base_cfg, load_path, client_root, lr)
                save_path, num_samples = client.train(self.base_cfg, load_path, client_root)
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

    # def _single_iter(self, load_path: Union[str, Path], edge_root: Path):
    #     weight_paths = []
    #     sample_counts = []

    #     num_gpus = 2
    #     gpu_ids = [0, 1]

    #     clients = list(self.clients)

    #     # Process clients in batches of size = num_gpus
    #     for batch_start in range(0, len(clients), num_gpus):
    #         batch = clients[batch_start : batch_start + num_gpus]

    #         processes = []
    #         manager = mp.Manager()
    #         results = manager.list()

    #         def _wrapped(client, gpu_id):
    #             client_root = edge_root / client.name
    #             client_root.mkdir(parents=True, exist_ok=True)
    #             save_path, num_samples = _run_client(
    #                 client, self.base_cfg, load_path, client_root, gpu_id
    #             )
    #             results.append((save_path, num_samples))

    #         # Launch up to 2 clients in parallel
    #         for client, gpu_id in zip(batch, gpu_ids):
    #             p = mp.Process(target=_wrapped, args=(client, gpu_id))
    #             p.start()
    #             processes.append(p)

    #         # Wait for this batch to finish
    #         for p in processes:
    #             p.join()
    #             if p.exitcode != 0:
    #                 raise RuntimeError("Client training process crashed")

    #         # Collect results from this batch
    #         for save_path, num_samples in results:
    #             weight_paths.append(save_path)
    #             sample_counts.append(num_samples)

    #     self.round += 1
    #     return weight_paths, sample_counts


    # def _single_iter(self, load_path: Union[str, Path], edge_root: Path):
    #     weight_paths = []
    #     sample_counts = []

    #     num_gpus = 2
    #     gpu_ids = [0, 1]
    #     clients = list(self.clients[:2])

    #     # manager = mp.Manager()
    #     # results = manager.list()
    #     result_queue = mp.Queue()

    #     # def _wrapped(client, gpu_id):
    #     #     client_root = edge_root / client.name
    #     #     client_root.mkdir(parents=True, exist_ok=True)
    #     #     save_path, num_samples = _run_client(
    #     #         client, self.base_cfg, load_path, client_root, gpu_id
    #     #     )
    #     #     results.append((save_path, num_samples))

    #     # Process clients in batches of size = num_gpus
    #     for batch_start in range(0, len(clients), num_gpus):
    #         batch = clients[batch_start: batch_start + num_gpus]
    #         processes = []

    #         # for client, gpu_id in zip(batch, gpu_ids):
    #         #     p = mp.Process(target=_wrapped, args=(client, gpu_id))
    #         #     p.start()
    #         #     processes.append(p)
    #         for client, gpu_id in zip(batch, gpu_ids):
    #             rounds = self.client_rounds[client.name]
    #             p = mp.Process(
    #                 target=_wrapped_worker,
    #                 args=(client, self.base_cfg, load_path, str(edge_root), gpu_id, rounds, result_queue)
    #             )
    #             p.start()
    #             processes.append(p)
    #             # pending_updates.append(client.name)
    #             self.client_rounds[client.name] += self.num_local_rounds

    #         for p in processes:
    #             p.join()
    #             if p.exitcode != 0:
    #                 raise RuntimeError("Client training process crashed")

    #         # for name in pending_updates:
    #         #     self.client_rounds[name] += self.num_local_rounds

    #     # for save_path, num_samples in results:
    #     #     weight_paths.append(save_path)
    #     #     sample_counts.append(num_samples)
    #     for _ in range(len(batch)):
    #         client_name, save_path_str, num_samples = result_queue.get()
    #         weight_paths.append(Path(save_path_str))
    #         sample_counts.append(num_samples)
    #         self.client_rounds[client_name] += self.num_local_rounds

    #     self.round += 1
    #     return weight_paths, sample_counts


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


