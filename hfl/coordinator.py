from hfl.aggregation import average_weights, save_state_dict
from hfl.edge import Edge

from typing import Optional, Union
from pathlib import Path
import json
import copy

import torch
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.apis import single_gpu_test
# from torch.utils.data import DataLoader
from mmdet.datasets import build_dataloader
from mmcv import print_log
from mmcv import Config

def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

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
                val_cfg_path: str,
                manifest_path: str,
                lr_cfg: dict,
                num_local_rounds: int = 1,
                num_edge_rounds: int = 1,
                num_global_rounds: int = 1,
                init_ckpt_path: Optional[str] = None,
                token_to_name_path: Optional[str] = None,
                seed: int = 0,
                resume_from: int = 0):

        if num_local_rounds <= 0: 
            raise ValueError("num_local_rounds must be a positive integer")
        if num_edge_rounds <= 0: 
            raise ValueError("num_edge_rounds must be a positive integer")
        if num_global_rounds <= 0: 
            raise ValueError("num_global_rounds must be a positive integer")

        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)

        self.resume_from = resume_from

        self.results_path = self.work_root / f"results.json"

        self.init_ckpt_path = init_ckpt_path
        self.num_global_rounds = num_global_rounds

        # self.results["global_rounds"] = []
        if self.resume_from == 0:
            self.results = {
                "global_rounds": []
            }
            self.start_from = 0
        else:
            with open(self.results_path, "r") as f:
                self.results = json.load(f)
            self.start_from = self.resume_from + 1

        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)
        
        print_log(f"Read manifest", logger='root' )

        # List of keys describing edges
        edges = self.manifest["edges"]

        base_cfg = Config.fromfile(base_cfg_path)
        self.val_cfg = Config.fromfile(val_cfg_path)
        print_log(f"Created base config file", logger='root' )

        # # Total number of client training epochs across complete federated training
        # # Necessary for some learning rate schedules
        # total_client_epochs = num_local_rounds * \
        #                       num_edge_rounds * \
        #                       num_global_rounds

        # From which epoch clients should resume from
        resume_from_clients = self.start_from*num_edge_rounds*num_local_rounds

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
                lr_cfg = lr_cfg,
                resume_from = resume_from_clients
            )
            self.edges.append(edge)


    def _load_model_weights(self, model, load_path: Union[str, Path]):
        load_path = Path(load_path)
        ckpt = torch.load(str(load_path), map_location="cpu")

        model_to_load = model.module if hasattr(model, "module") else model

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            print(f"Using MMCV load_checkpoint for MMDet checkpoint: {load_path}", flush=True)
            load_checkpoint(model_to_load, str(load_path), map_location="cpu", strict=False)
            return

        state_dict = ckpt

        # Build name -> actual tensor reference maps
        name_to_param = dict(model_to_load.named_parameters())
        name_to_buf = dict(model_to_load.named_buffers())

        loaded = 0
        skipped = 0
        shape_mismatch = 0

        for k, v in state_dict.items():
            target = None
            if k in name_to_param:
                target = name_to_param[k]
            elif k in name_to_buf:
                target = name_to_buf[k]

            if target is None:
                skipped += 1
                continue

            if target.shape != v.shape:
                shape_mismatch += 1
                # print first few mismatches only
                if shape_mismatch <= 5:
                    print(f"SHAPE MISMATCH {k}: ckpt={tuple(v.shape)} model={tuple(target.shape)}", flush=True)
                continue

            # Copy weights WITHOUT triggering module-specific load logic
            target.data.copy_(v)
            loaded += 1

        print(f"Manual load complete. loaded={loaded} skipped_missing={skipped} skipped_shape={shape_mismatch}", flush=True)


    def _single_iter(self, load_path: Union[str, Path], global_root):
        weight_paths = []
        sample_counts = []
        edge_results = []
        for edge in self.edges:
            edge_root = global_root / str(edge.name)
            edge_root.mkdir(parents=True, exist_ok=True)

            # save_path, num_samples = edge.train(load_path, edge_root)
            save_path, num_samples, train_results = edge.train(load_path, edge_root)

            edge_results.append(train_results)

            weight_paths.append(save_path)
            sample_counts.append(num_samples)

        # return weight_paths, sample_counts
        return weight_paths, sample_counts, edge_results


    def train(self):
        if self.resume_from == 0:
            load_path = self.init_ckpt_path
        else:
            load_path = self.work_root / f"global_round_{self.resume_from}" / "global_weights.pth"

        for i in range(self.start_from, self.num_global_rounds):
            print_log(f"[CLOUD] - Round {i}", logger='root' )
            global_root = self.work_root / f"global_round_{i}"
            global_path = global_root / "global_weights.pth"
            global_root.mkdir(parents=True, exist_ok=True)

            # Train across all edges
            # weight_paths, sample_counts = self._single_iter(load_path, global_root)
            weight_paths, sample_counts, train_results = self._single_iter(load_path, global_root)

            # Aggregate edge weights
            avg_weights = average_weights(weight_paths, sample_counts)

            # Store aggregated weights
            save_state_dict(avg_weights, global_path)
            load_path = global_path

            val_results = self.validate(global_path)
            edge_results = {"global_round": i, "edges": train_results, "val_results": val_results}
            self.results["global_rounds"].append(edge_results)

            write_json(self.results_path, self.results)


    def validate(self, weights_path):
        print_log(f"validating weights located at {weights_path}", logger='root' )

        cfg = copy.deepcopy(self.val_cfg)
        dataset = build_dataset(cfg.data.val)

        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False
        )
        
        model = build_model(
            cfg.model,
            train_cfg=None,
            test_cfg=cfg.get('test_cfg')
        )
        model.init_weights()
        model.CLASSES = dataset.CLASSES
        self._load_model_weights(model, weights_path)
        
        model.eval()
        model = MMDataParallel(model, device_ids=[0])

        with torch.no_grad():
            outputs = single_gpu_test(model, data_loader)

        metrics = dataset.evaluate(
            outputs,
            metric=cfg.evaluation.metric
        )

        print_log(f"VALIDATION RESULTS: {metrics}", logger='root')
        return metrics
