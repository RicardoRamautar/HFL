from pathlib import Path
import copy
import json
import gc

import torch
from mmcv import Config

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.apis.train import train_detector


class HFL:
    """Hierarchical FL orchestrator for CMT(-BTSA) on nuScenes."""

    def __init__(
        self,
        work_root: str,
        base_config: str,
        data_manifest: str,
        initial_ckpts: str,
        total_rounds: int = 1,
        nr_edge_rounds: int = 1,
        log: bool = False,
        scene_translation: str | None = None,
    ):
        self.work_root = Path(work_root)
        self.base_config = base_config
        self.data_manifest = data_manifest
        self.initial_ckpts = initial_ckpts
        self.total_rounds = total_rounds
        self.nr_edge_rounds = nr_edge_rounds
        self.log = log
        self.scene_translation = scene_translation

        # base cfg shared by all clients (architecture + hyperparams)
        self.base_cfg = Config.fromfile(self.base_config)

        with open(self.data_manifest, "r") as f:
            self.manifest = json.load(f)

        self.global_round = 0
        self.edge_round = 0

        self.work_root.mkdir(parents=True, exist_ok=True)

    def _build_client_cfg(self, scenes, work_dir: Path) -> Config:
        """Construct a client-specific config derived from base config."""
        cfg = copy.deepcopy(self.base_cfg)

        cfg.work_dir = str(work_dir)

        # Client-specific dataset slicing
        cfg.data.train.enable_hfl = True
        cfg.data.train.scenes = scenes
        if self.scene_translation is not None:
            cfg.data.train.scene_translation = self.scene_translation

        # FL: avoid costly checkpointing + avoid validation per client
        cfg.checkpoint_config = None
        cfg.evaluation = None

        # Avoid MMDet implicit checkpoint loading
        cfg.load_from = None
        cfg.resume_from = None

        # (optional) reduce worker overhead
        cfg.data.workers_per_gpu = min(int(cfg.data.get("workers_per_gpu", 6)), 2)

        return cfg

    def _load_model_weights(self, model, weights_path: Path):
        """Load model-only weights into model."""
        sd = torch.load(str(weights_path), map_location="cpu")
        model_to_load = model.module if hasattr(model, "module") else model
        model_to_load.load_state_dict(sd, strict=True)

    def _save_model_weights(self, model, weights_path: Path):
        """Save model-only weights."""
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), str(weights_path))

    def _train_client(self, cfg: Config, in_weights: Path, out_weights: Path):
        """Train one client locally and write model-only weights."""
        dataset = build_dataset(cfg.data.train)

        model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
        model.init_weights()

        # Load incoming (global/edge) weights
        self._load_model_weights(model, in_weights)

        model.CLASSES = dataset.CLASSES

        train_detector(model, [dataset], cfg, distributed=False, validate=False)

        # Save outgoing client update
        self._save_model_weights(model, out_weights)

        if self.log:
            print(f"[HFL] Saved client weights: {out_weights}")

    def _edge_iteration(self, clients: dict, edge_name: str, in_edge_weights: Path, out_dir: Path):
        """Train all clients on one edge for current (global_round, edge_round)."""
        for client_id, client in clients.items():
            scenes = client["scenes"]

            client_dir = out_dir / f"client_{client_id}"
            client_dir.mkdir(parents=True, exist_ok=True)

            # Round-aware outputs
            client_work_dir = client_dir / f"g{self.global_round}" / f"e{self.edge_round}"
            client_work_dir.mkdir(parents=True, exist_ok=True)

            out_client_weights = client_work_dir / f"client_{client_id}_g{self.global_round}_e{self.edge_round}.pth"

            if self.log:
                print(f"\n=== Train edge={edge_name} client={client_id} g={self.global_round} e={self.edge_round} ===")
                print(f"Scenes: {scenes}")
                print(f"Work dir: {client_work_dir}")
                print(f"IN : {in_edge_weights}")
                print(f"OUT: {out_client_weights}")

            cfg = self._build_client_cfg(scenes=scenes, work_dir=client_work_dir)

            try:
                self._train_client(cfg, in_weights=in_edge_weights, out_weights=out_client_weights)
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _global_iteration(self):
        """Single global round consisting of nr_edge_rounds local edge cycles (aggregation not implemented here)."""
        edges = self.manifest["edges"]

        # For now: define the incoming weights for this global round
        # Round 0 incoming = initial_ckpts; later rounds should use global aggregated weights
        if self.global_round == 0:
            in_global_weights = Path(self.initial_ckpts)
        else:
            in_global_weights = self.work_root / f"global_g{self.global_round}.pth"  # placeholder

        for e in range(self.nr_edge_rounds):
            self.edge_round = e

            for edge_name, edge_data in edges.items():
                clients = edge_data["clients"]

                edge_dir = self.work_root / edge_name
                edge_dir.mkdir(parents=True, exist_ok=True)

                # Incoming edge model (for now, same as global model; later: edge model)
                in_edge_weights = in_global_weights

                # Run clients on this edge
                self._edge_iteration(
                    clients=clients,
                    edge_name=edge_name,
                    in_edge_weights=in_edge_weights,
                    out_dir=edge_dir,
                )

            # After all edges complete, you would do edge aggregation here
            # -> produce edge_{edge}_g{g}_e{e+1}.pth
            # Not implemented in this skeleton.

        self.edge_round = 0

    def train(self):
        for g in range(self.total_rounds):
            self.global_round = g
            self._global_iteration()

            # After edge cycles, you would do global aggregation here
            # -> produce global_g{g+1}.pth
            # Not implemented in this skeleton.
