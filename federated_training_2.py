from typing import Optional, Union
from pathlib import Path
import copy
import json
import gc

from mmcv import Config
import torch

from mmdet3d.apis.train import train_detector
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

class Client():
    def __init__(self, 
                 cid,
                 scenes,
                 base_cfg,
                 workdir: str,
                 num_epochs: Optional[int] = None,
                 seed = 0,
                 scene_translation_path = None):

        self.cid = cid
        self.workdir = workdir

        self.cfg = self._build_client_cfg(
            base_cfg,
            scenes,
            scene_translation_path,
            seed,
            num_epochs
        )
        self.meta = self._init_runner_meta()


    def _build_client_cfg(self, base_cfg, scenes, scene_translation_path, seed, num_epochs):
        """ Construct client-specific config from base config."""
        cfg = copy.deepcopy(base_cfg)

        # Overwrite client-specific information
        cfg.work_dir = str(self.work_dir)

        cfg.data.train.enable_hfl = True
        cfg.data.train.scenes = scenes

        if scene_translation_path is not None:
            cfg.data.train.scene_translation = scene_translation_path
        cfg.evaluation = None
        cfg.checkpoint_config = None
        cfg.seed = seed

        if num_epochs is not None:
            cfg.total_epochs = num_epochs

        cfg.gpu_ids = [0]

        # We set the model weights ourselves
        cfg.load_from = None
        cfg.resume_from = None

        return cfg


    def _init_runner_meta(self):
        """Initialize runner.meta similar to tools/train.py (minimal subset)."""
        meta = {}

        # Seed handling (required)
        seed = getattr(self.cfg, "seed", 0)
        cfg.seed = seed
        meta["seed"] = seed

        # Optional but useful
        meta["exp_name"] = "hfl_client"
        meta["config"] = cfg.pretty_text

        return meta


    def _load_model_weights(self, model, weights_path: Path):
        """ Load model-only weights."""
        ckpt = torch.load(str(weights_path), map_location="cpu")

        # Support model-only weights and MMDET cktps
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        model_to_load = model.module if hasattr(model, "module") else model
        model_to_load.load_state_dict(state_dict, strict=False)


    def _save_model_weights(self, model, weights_path: Path):
        """Store model-only weights."""
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), str(weights_path))


    def train(self, root, in_weights_path):
        out_weights_path = root / f"{self.cid}"
        out_weights_path.mkdir(parents=True, exist_ok=True)

        # Construct dataset
        dataset =  build_dataset(self.cfg.data.train)
        num_samples = len(dataset)

        # Build model
        model = build_model(
            self.cfg.model,
            train_cfg=self.cfg.get('train_cfg'),
            test_cfg=self.cfg.get('test_cfg')
        )
        model.init_weights()

        # Make class names available
        model.CLASSES = dataset.CLASSES

        # Load model weights
        self._load_model_weights(model, in_weights_path)

        # Train
        train_detector(
            model,
            [dataset],
            self.cfg,
            distributed=False,
            validate=False,
            meta = self.meta
        )

        # Store weights
        self._save_model_weights(model, out_weights_path)
        
        print(f"Saved client weights: {out_weights}")

        return (out_weights_path, num_samples)



class Edge():
    def __init__(self, 
                 name,
                 clients,
                 num_rounds,
                 base_cfg,
                 work_root,
                 seed,
                 scene_translation_path,
                 num_epochs: Optional[int] = None):

        self.name = name
        self.num_rounds = num_rounds
        self.base_cfg = base_cfg
        self.work_root = work_root

        # self.clients = [Client(cid,scenes,) cid, scenes in clients.items()]
        self.clients = []
        for cid, scenes in clients.items():
            client = Client(
                cid,
                scenes,
                self.base_cfg
                self.work_root
                seed
                scene_translation_path
            )
            self.clients.append(client)

        # Store paths were trainign weights and .json are stored for every iteration
        self.prev_weights_path = None

        self.round = 0

    def _init_round(self, work_root):
        # Create root directory of new global training round
        root = work_root / f"edge_round_{self.round}"
        root.mkdir(parents=True, exist_ok=True)

        out_weights_path = root / "edge_model.pth"

        return (root, out_weights_path)


    def _average_weights(self, weight_paths, sample_counts):
        total_samples = sum(sample_counts)
        if total_samples == 0:
            raise ValueError(f"Cannot average over 0 training samples")

        avg_weights = None
        for path, samples in zip(weight_paths, sample_counts):
            ckpt = torch.load(str(path), map_location="cpu")
            state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

            w = samples / total_samples

            if avg_weights is None:
                avg_weights = {k: v * w for k, v in state_dict.items()}
            else:
                for k in avg_weights:
                    avg_weights[k] += state_dict[k] * w

        return avg_weights, total_samples


    def _save_state_dict(self, state_dict: dict, weights_path: Path):
        """Store a state_dict directly (used for aggregated models)."""
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, str(weights_path))


    def _single_iter(self, client_root, in_weights_path, out_weights_path):
        weight_paths = []
        sample_counts = []
        for client in self.clients:
            client_weight_path, num_samples = client.train(client_root, in_weights_path)
            weight_paths.append(client_weight_path)
            sample_counts.append(num_samples)

        avg_weights, total_samples = self._average_weights(weight_paths, sample_counts)
        self._save_state_dict(avg_weights, out_weights_path)

        return total_samples


    def train(self, root, in_weights_path):
        total_samples = 0
        for i in range(self.num_rounds):
            self.round = i 

            client_root, out_weights_path = self._init_round(root)
            num_samples = self._single_iter(client_root, in_weights_path, out_weights_path)

            in_weights_path = out_weights_path
            total_samples += num_samples

        return out_weights_path, total_samples




class Coordinator():
    def __init__(self,
                 work_root: str, 
                 base_config_path: str,
                 data_assignment_path: str,
                 initial_ckpts_path: str,
                 num_epochs: Optional[int] = None,
                 num_global_rounds: int = 1,
                 num_edge_rounds: int = 1,
                 scene_translation_path: Optional[str] = None,
                 log: bool = False,
                 seed: int = 0):

        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)

        self.base_config_path = base_config_path
        self.data_assignment_path = data_assignment_path
        self.initial_ckpts_path = initial_ckpts_path
        self.scene_translation_path = scene_translation_path

        self.num_global_rounds = num_global_rounds
        
        self.log = log
        self.seed = seed

        # Load Dict that assigns scenes to clients
        with open(self.data_assignment_path, "r") as f:
            self.data_assignment = json.load(f)

        # Initialize edges
        self.edges = []
        for edge_name, edge_data in self.data_assignment.items():
            edge = Edge(
                name = edge_name,
                clients = edge_data["clients"],
                num_edge_rounds,
                num_epochs = num_epochs,
                base_cfg = self.base_config_path,
                work_root,
                seed,
                scene_translation_path,
            )
            self.edges.append(edge)

        self.round = 0
        self.global_paths = []

    def _init_round(self):
        # Create root directory of new global training round
        root = self.work_root / f"global_round_{self.round}"
        root.mkdir(parents=True, exist_ok=True)

        out_weights_path = root / "global_model.pth"

        return (root, out_weights_path)


    def _average_weights(self, weight_paths, sample_counts):
        total_samples = sum(sample_counts)
        if total_samples == 0:
            raise ValueError(f"Cannot average over 0 training samples")

        avg_weights = None
        for path, samples in zip(weight_paths, sample_counts):
            ckpt = torch.load(str(path), map_location="cpu")
            state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

            w = samples / total_samples

            if avg_weights is None:
                avg_weights = {k: v * w for k, v in state_dict.items()}
            else:
                for k in avg_weights:
                    avg_weights[k] += state_dict[k] * w

        return avg_weights, total_samples


    def _save_state_dict(self, state_dict: dict, weights_path: Path):
        """Store a state_dict directly (used for aggregated models)."""
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, str(weights_path))


    def _single_iter(self, root, in_weights_path, out_weights_path):
        weight_paths = []
        sample_counts = []
        for edge in self.edges:
            edge_weights_path, edge_sample_count = edge.train(root, in_weights_path)

            self.weight_paths.append(edge_weights_path)
            self.sample_counts.append(edge_sample_count)

        avg_weights, total_samples = self._average_weights(weight_paths, sample_counts)
        self._save_state_dict(avg_weights, out_weights_path)


    def train(self):
        """Perform global iterations."""

        for i in range(self.num_global_rounds):
            self.round = i 

            if self.round == 0:
                in_weights_path = self.initial_ckpts_path

            edge_root, out_weights_path = self._init_round()
            self._single_iter(edge_root, in_weights_path, out_weights_path)

            in_weights_path = out_weights_path

        return




# global_round_1
#     - edge_round_1
#         - edge_1
#             - client_01
#             - client_02
#             - ...
#         - edge_2
#         - ...
#     - edge_round_2
#         - ...
#     - ...
#     - global_model.pth