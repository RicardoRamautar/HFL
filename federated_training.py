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

class HFL():
    """ Implementation of HierFAVG for training CMT on nuScenes.

    Args:
        work_root (str): Root path where training results will be stored
        base_config_path (str): File path to location of the CMT's base config file for clients
        data_assignment_path (str): File path to json file containing assignming of nuScenes
            scenes to clients and clients to edges
        initial_ckpts_path (str): File path to initial model weights
        total_rounds (int, Default=1): Total number of global training rounds
        nr_edge_rounds (int, Default=1): Number of edge aggregation rounds within a single
            global training round.
        log (bool, Default=False): Whether to print out logs
        scene_translation_path (str|None, Default=None): File path to json file containing
            mapping from scene token to scene name if not already provided in base config
    """
    def __init__(self,
                 work_root: str, 
                 base_config_path: str,
                 data_assignment_path: str,
                 initial_ckpts_path: str,
                 total_rounds: int = 1,
                 nr_edge_rounds: int = 1,
                 log: bool = False,
                 scene_translation_path: Optional[str] = None,
                 seed: int = 0):

        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)

        self.base_config_path = base_config_path
        self.data_assignment_path = data_assignment_path
        self.initial_ckpts_path = initial_ckpts_path
        self.total_rounds = total_rounds
        self.nr_edge_rounds = nr_edge_rounds
        self.log = log
        self.scene_translation_path = scene_translation_path
        self.seed = seed

        #  Build base config used by all clients
        self.base_cfg = Config.fromfile(base_config_path)

        # Load Dict that assigns scenes to clients
        with open(self.data_assignment_path, "r") as f:
            self.data_assignment = json.load(f)

        self.global_round = 0
        self.edge_round = 0


    def _init_runner_meta(self, cfg: Config):
        """Initialize runner.meta similar to tools/train.py (minimal subset)."""
        meta = {}

        # Seed handling (required)
        seed = getattr(cfg, "seed", 0)
        cfg.seed = seed
        meta["seed"] = seed

        # Optional but useful
        meta["exp_name"] = "hfl_client"
        meta["config"] = cfg.pretty_text

        return meta


    def _build_client_cfg(self, scenes, work_dir: Union[Path, str]):
        """ Construct client-specific config from base config."""
        cfg = copy.deepcopy(self.base_cfg)

        # Overwrite client-specific information
        cfg.work_dir = str(work_dir)
        cfg.data.train.enable_hfl = True
        cfg.data.train.scenes = scenes
        if self.scene_translation_path is not None:
            cfg.data.train.scene_translation = self.scene_translation_path
        cfg.evaluation = None
        cfg.checkpoint_config = None
        cfg.seed = self.seed

        cfg.gpu_ids = [0]

        # We set the model weights ourselves
        cfg.load_from = None
        cfg.resume_from = None

        return cfg


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

    def _save_state_dict(self, state_dict: dict, weights_path: Path):
        """Store a state_dict directly (used for aggregated models)."""
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, str(weights_path))

    def _train_client(self, cfg: Config, in_weights: Path, out_weights: Path):
        """ Train client network. """
        # Construct client dataset
        dataset = build_dataset(cfg.data.train)

        model = build_model(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )
        model.init_weights()

        self._load_model_weights(model, in_weights)
        
        # Make class names available
        model.CLASSES = dataset.CLASSES

        # Train
        train_detector(
            model,
            [dataset],
            cfg,
            distributed=False,
            validate=False,
            meta = self._init_runner_meta(cfg)
        )

        # Store weights
        self._save_model_weights(model, out_weights)
        if self.log:
            print(f"Saved client weights: {out_weights}")

        return len(dataset)


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

        return avg_weights


    def _edge_iteration(self, clients: dict, edge_name: str, in_edge_weights: Path, out_dir: Path):
        """ Single training iteration across edge server {edge_name}."""
        weight_paths = []
        sample_counts = []

        n_clients = {k: clients[k] for k in list(clients.keys())[:5]}
        for client_id, client in n_clients.items():
            # nuScenes scenes assigned to client
            scenes = client["scenes"]

            client_dir = out_dir / f"{client_id}"
            client_dir.mkdir(parents=True, exist_ok=True)

            client_work_dir = client_dir / f"g{self.global_round}" / f"e{self.edge_round}"
            client_work_dir.mkdir(parents=True, exist_ok=True)

            out_client_weights = client_work_dir / f"{client_id}_g{self.global_round}_e{self.edge_round}.pth"
            weight_paths.append(out_client_weights)

            if self.log:
                print(f"\n=== Train edge={edge_name} client={client_id} g={self.global_round} e={self.edge_round} ===")
                print(f"Scenes: {scenes}")
                print(f"Work dir: {client_work_dir}")
                print(f"IN : {in_edge_weights}")
                print(f"OUT: {out_client_weights}")

            cfg = self._build_client_cfg(
                scenes=scenes,
                work_dir=str(client_work_dir),
            )

            try:
                sample_count = self._train_client(
                    cfg, 
                    in_weights=in_edge_weights, 
                    out_weights=out_client_weights
                )
                sample_counts.append(sample_count)
            finally:
                # Free Python objects and cached GPU memory between clients
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return self._average_weights(weight_paths, sample_counts)


    def _global_iteration(self):
        """Single global training iteration."""

        get_global_path = lambda g : self.work_root / f"global_g{g}.pth"
        # get_edge_path = lambda g,e,i : self.work_root / f"g_{g}" / f"{e}" / f"round_{i}"
        get_edge_path = lambda g, edge, r: (
            self.work_root / edge / f"g{g}" / f"e{r}" / "edge.pth"
        )


        out_global_weights = get_global_path(self.global_round+1)
        if self.global_round == 0:
            in_global_weights = Path(self.initial_ckpts_path)
        else:
            # in_global_weights = self.work_root / f"global_g{self.global_round}.pth"
            in_global_weights = get_global_path(self.global_round)


        # List of keys describing edges
        edges = self.data_assignment["edges"]

        for e in range(self.nr_edge_rounds):
            self.edge_round = e

            # Iterate over all edge servers
            for edge_name, edge_data in edges.items():
                out_edge_weights = get_edge_path(self.global_round, edge_name, e+1)
                out_edge_weights.parent.mkdir(parents=True, exist_ok=True)
                if e == 0:
                    in_edge_weights = in_global_weights
                else:
                    in_edge_weights = get_edge_path(self.global_round, edge_name, e)

                # Get clients assigned to edge server
                clients = edge_data["clients"]

                edge_dir = self.work_root / edge_name
                edge_dir.mkdir(parents=True, exist_ok=True)

                # Perform single edge training round
                edge_weights = self._edge_iteration(
                    clients= clients, 
                    edge_name= edge_name,
                    in_edge_weights= in_edge_weights,
                    out_dir = edge_dir
                )

                self._save_state_dict(edge_weights, out_edge_weights)

        weight_paths = []
        num_samples = []
        for edge_name in edges.keys():
            out_edge_weights = get_edge_path(self.global_round, edge_name, self.nr_edge_rounds)
            weight_paths.append(out_edge_weights)
            num_samples.append(1)       # For now, do not weight edge ckpts, we implement that later
        avg_weights = self._average_weights(weight_paths, num_samples)
        self._save_state_dict(avg_weights, out_global_weights)


    def train(self):
        for g in range(self.total_rounds):
            self.global_round = g
            self._global_iteration()

        return            

def main():
    hfl = HFL(
        work_root = "/tudelft.net/staff-umbrella/rdramautar/CMT-BTSA/workdirs/federated_test_3/",
        base_config_path = "/tudelft.net/staff-umbrella/rdramautar/CMT-BTSA/configs/federated/cmt_lidar_voxel0075_cbgs.py",
        data_assignment_path = "/tudelft.net/staff-umbrella/rdramautar/HFMTL/dataset_distribution.json",
        initial_ckpts_path = "/tudelft.net/staff-umbrella/rdramautar/ckpts/lidar_voxel0075_epoch20.pth",
        total_rounds = 1,
        nr_edge_rounds = 1,
        log = True,
        scene_translation_path = "/tudelft.net/staff-umbrella/rdramautar/HFMTL/scene_name_to_token.json"
    )
    hfl.train()


if __name__=="__main__":
    main()