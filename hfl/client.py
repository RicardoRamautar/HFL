from typing import List, Optional, Union
from pathlib import Path
import copy

from mmcv import Config
import torch

from mmdet3d.apis.train import train_detector
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from mmcv import print_log


class Client():
    """ Client participating in federated learning.

    Args:
        name (str): Unique client id.
        scenes(List[str]): List of nuScenes scene names assigned to this client.
        base_cfg_path (str): File path to model config template.
        num_epochs (int): Number of training epochs.
            Default: 1
        token_to_name_path (Optional[str]): File path to json storing mapping from
            nuScenes scene name and scene token.
        seed (int): Training seed for deterministic training.
            Default: 0
    """
    def __init__(self,
                 name: str,
                 scenes:List[str],
                 num_epochs: int = 1,
                 token_to_name_path: Optional[str] = None,
                 seed: int = 0):

        if num_epochs <= 0:
            raise ValueError(f"Number of training epochs must be a positive integer.")

        self.name = name
        self.scenes = scenes
        self.num_epochs = num_epochs
        self.token_to_name_path = token_to_name_path
        self.seed = seed
        self.num_samples = 0


    def _build_client_cfg(self, 
                          scenes: List[str], 
                          base_cfg: Config, 
                          num_epochs: int, 
                          token_to_name_path: Union[str, None], 
                          seed: int,
                          lr: float):
        """ Construct client-specific config from base config.
        
        Args:
            scenes(List[str]): List of nuScenes scene names assigned to this client.
            base_cfg (Config): Client-agnostic model config.
            num_epochs (int): Number of training epochs.
                Default: 1
            token_to_name_path (Optional[str]): File path to json storing mapping from
                nuScenes scene name and scene token.
            seed (int): Training seed for deterministic training.
                Default: 0

        """
        cfg = copy.deepcopy(base_cfg)

        # Overwrite client-specific information
        cfg.data.train.enable_hfl = True
        cfg.data.train.scenes = scenes

        if token_to_name_path is not None:
            cfg.data.train.scene_translation = token_to_name_path
        cfg.evaluation = None
        cfg.checkpoint_config = None
        cfg.seed = seed

        if num_epochs is not None:
            cfg.total_epochs = num_epochs
            cfg.runner.max_epochs = num_epochs

        cfg.gpu_ids = [0]

        # We set the model weights ourselves
        cfg.load_from = None
        cfg.resume_from = None

        cfg.lr_config = None
        cfg.momentum_config = None
        cfg.optimizer.lr = lr

        return cfg


    def _init_runner_meta(self, cfg):
        """Minimal initialization of runner.meta similar to tools/train.py."""
        meta = {}

        # Seed handling (required)
        meta["seed"] = cfg.seed

        # Optional but useful
        meta["exp_name"] = "hfl_client"
        meta["config"] = cfg.pretty_text

        return meta


    # def _load_model_weights(self, model, load_path: Union[str, Path]):
    #     """ Load model weights from file."""
    #     ckpt = torch.load(str(load_path), map_location="cpu")

    #     # Support model-only weights and MMDET cktps
    #     state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    #     model_to_load = model.module if hasattr(model, "module") else model
    #     model_to_load.load_state_dict(state_dict, strict=False)

    # def _load_model_weights(self, model, load_path: Union[str, Path]):
    #     ckpt = torch.load(str(load_path), map_location="cpu")

    #     # Unwrap checkpoint
    #     if isinstance(ckpt, dict) and "state_dict" in ckpt:
    #         state_dict = ckpt["state_dict"]
    #         print_log("Loaded MMDet-style checkpoint (contains state_dict)", logger="root")
    #     else:
    #         state_dict = ckpt
    #         print_log("Loaded raw state_dict", logger="root")

    #     # Basic sanity
    #     print_log(f"Number of tensors in state_dict: {len(state_dict)}", logger="root")

    #     # Inspect ONE representative sparse-conv key
    #     key = "pts_middle_encoder.conv_input.0.weight"
    #     if key in state_dict:
    #         w = state_dict[key]
    #         print_log(
    #             f"{key} shape in checkpoint: {tuple(w.shape)}",
    #             logger="root"
    #         )
    #     else:
    #         print_log(
    #             f"{key} NOT FOUND in checkpoint",
    #             logger="root"
    #         )

    #     model_to_load = model.module if hasattr(model, "module") else model
    #     model_sd = model_to_load.state_dict()

    #     if key in model_sd:
    #         print_log(
    #             f"{key} shape in model: {tuple(model_sd[key].shape)}",
    #             logger="root"
    #         )

    #     # Try loading
    #     model_to_load.load_state_dict(state_dict, strict=False)

    def _load_model_weights(self, model, load_path: Union[str, Path]):
        from pathlib import Path
        import torch
        from mmcv.runner import load_checkpoint

        load_path = Path(load_path)
        ckpt = torch.load(str(load_path), map_location="cpu")

        model_to_load = model.module if hasattr(model, "module") else model

        # 1) MMDet checkpoint: always use MMCV loader (handles spconv formats correctly)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            print(f"[{self.name}] Using MMCV load_checkpoint for MMDet checkpoint: {load_path}", flush=True)
            load_checkpoint(model_to_load, str(load_path), map_location="cpu", strict=False)
            return

        # 2) Raw state_dict (your FL weights): DO NOT call load_state_dict()
        print(f"[{self.name}] Loading raw state_dict via MANUAL COPY: {load_path}", flush=True)
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
                    print(f"[{self.name}] SHAPE MISMATCH {k}: ckpt={tuple(v.shape)} model={tuple(target.shape)}", flush=True)
                continue

            # Copy weights WITHOUT triggering module-specific load logic
            target.data.copy_(v)
            loaded += 1

        print(f"[{self.name}] Manual load complete. loaded={loaded} skipped_missing={skipped} skipped_shape={shape_mismatch}", flush=True)



    def _save_model_weights(self, model, save_path: Union[str, Path]):
        """Store model-only weights."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), str(save_path))


    def train(self, 
              base_cfg: Config,
              load_path: Union[str, Path], 
              work_root: Path,
              lr: float):
        """ Train model on client dataset.

        Args:
            load_path (str | Path): :File path containing initial weights.
            root (Path): Folder path where the training results will be stored.
            save_path (Path): File path where updated weights will be stored.
            round_idx (int): How many edge aggregations rounds have been performed.
        """
        cfg = self._build_client_cfg(
            scenes = self.scenes,
            base_cfg = base_cfg,
            num_epochs = self.num_epochs,
            token_to_name_path = self.token_to_name_path,
            seed = self.seed,
            lr = lr     
        )
        meta = self._init_runner_meta(cfg)


        print_log(f"[CLIENT - {self.name}] has started training")

        cfg.work_dir = str(work_root)
        save_path = work_root / "weights.pth"

        # Construct dataset
        dataset =  build_dataset(cfg.data.train)
        num_samples = len(dataset)

        # Build model
        model = build_model(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )
        model.init_weights()

        # Make class names available
        model.CLASSES = dataset.CLASSES

        # Load model weights
        self._load_model_weights(model, load_path)

        # Train
        train_detector(
            model,
            [dataset],
            cfg,
            distributed=False,
            validate=False,
            meta = meta
        )
        print(f"Client {self.name} has finished training.")

        # Store weights
        self._save_model_weights(model, save_path)
        
        print(f"Client {self.name} stored weights at {str(save_path)}")
        return save_path, num_samples


# from typing import List, Optional, Union
# from pathlib import Path
# import copy
# import os
# import hashlib
# import logging

# # 🔇 KILL ALL LOGGING — DO THIS BEFORE MMCV IMPORTS
# logging.disable(logging.CRITICAL)

# from mmcv import Config
# import torch

# from mmdet3d.apis.train import train_detector
# from mmdet3d.datasets import build_dataset
# from mmdet3d.models import build_model


# class Client():
#     def __init__(self,
#                  name: str,
#                  scenes: List[str],
#                  num_epochs: int = 1,
#                  token_to_name_path: Optional[str] = None,
#                  seed: int = 0):

#         if num_epochs <= 0:
#             raise ValueError("num_epochs must be > 0")

#         self.name = name
#         self.scenes = scenes
#         self.num_epochs = num_epochs
#         self.token_to_name_path = token_to_name_path
#         self.seed = seed
#         self.num_samples = 0

#     # ------------------------------------------------------------------
#     # CONFIG
#     # ------------------------------------------------------------------
#     def _build_client_cfg(self, scenes, base_cfg, num_epochs, token_to_name_path, seed, lr):
#         cfg = copy.deepcopy(base_cfg)

#         cfg.data.train.enable_hfl = True
#         cfg.data.train.scenes = scenes
#         if token_to_name_path is not None:
#             cfg.data.train.scene_translation = token_to_name_path

#         cfg.evaluation = None
#         cfg.checkpoint_config = None
#         cfg.seed = seed
#         cfg.total_epochs = num_epochs
#         cfg.gpu_ids = [0]

#         cfg.load_from = None
#         cfg.resume_from = None

#         cfg.lr_config = None
#         cfg.momentum_config = None
#         cfg.optimizer.lr = lr

#         return cfg

#     def _init_runner_meta(self, cfg):
#         return {
#             "seed": cfg.seed,
#             "exp_name": "hfl_client",
#             "config": cfg.pretty_text,
#         }

#     # ------------------------------------------------------------------
#     # WEIGHT LOADING — HARD DIAGNOSTICS
#     # ------------------------------------------------------------------
#     # def _load_model_weights(self, model, load_path: Union[str, Path]):
#     #     load_path = Path(load_path)

#     #     def md5(path, nbytes=2_000_000):
#     #         with open(path, "rb") as f:
#     #             return hashlib.md5(f.read(nbytes)).hexdigest()

#     #     pid = os.getpid()
#     #     try:
#     #         import torch.distributed as dist
#     #         rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
#     #     except Exception:
#     #         rank = 0

#     #     print("\n================ LOAD MODEL WEIGHTS ================", flush=True)
#     #     print(f"[CLIENT={self.name}] PID={pid} RANK={rank}", flush=True)
#     #     print(f"Loading file: {load_path}", flush=True)
#     #     print(f"MD5: {md5(load_path)}", flush=True)

#     #     ckpt = torch.load(str(load_path), map_location="cpu")

#     #     if isinstance(ckpt, dict) and "state_dict" in ckpt:
#     #         state_dict = ckpt["state_dict"]
#     #         print("Checkpoint type: MMDet-style (state_dict key)", flush=True)
#     #     else:
#     #         state_dict = ckpt
#     #         print("Checkpoint type: raw state_dict", flush=True)

#     #     print(f"Number of tensors in checkpoint: {len(state_dict)}", flush=True)

#     #     model_to_load = model.module if hasattr(model, "module") else model
#     #     model_sd = model_to_load.state_dict()

#     #     # Representative sparse-conv key
#     #     probe_key = "pts_middle_encoder.conv_input.0.weight"

#     #     if probe_key in state_dict and probe_key in model_sd:
#     #         print(
#     #             f"PROBE CKPT {probe_key}: {tuple(state_dict[probe_key].shape)}",
#     #             flush=True
#     #         )
#     #         print(
#     #             f"PROBE MODEL {probe_key}: {tuple(model_sd[probe_key].shape)}",
#     #             flush=True
#     #         )
#     #     else:
#     #         print(f"PROBE KEY MISSING: {probe_key}", flush=True)

#     #     # Detect mismatches
#     #     mismatches = []
#     #     for k, v in state_dict.items():
#     #         if k in model_sd and v.shape != model_sd[k].shape:
#     #             mismatches.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
#     #             if len(mismatches) >= 5:
#     #                 break

#     #     print(f"Mismatched tensors: {len(mismatches)}", flush=True)
#     #     for k, a, b in mismatches:
#     #         print(f"  {k}: ckpt={a} model={b}", flush=True)

#     #     # SAFE LOAD: only matching tensors
#     #     filtered = {
#     #         k: v for k, v in state_dict.items()
#     #         if k in model_sd and v.shape == model_sd[k].shape
#     #     }

#     #     print(
#     #         f"Loading {len(filtered)} tensors, skipping {len(state_dict) - len(filtered)}",
#     #         flush=True
#     #     )

#     #     model_to_load.load_state_dict(filtered, strict=False)
#     #     print("LOAD COMPLETE\n", flush=True)

#     def _load_model_weights(self, model, load_path: Union[str, Path]):
#         from pathlib import Path
#         import torch
#         from mmcv.runner import load_checkpoint

#         load_path = Path(load_path)
#         ckpt = torch.load(str(load_path), map_location="cpu")

#         model_to_load = model.module if hasattr(model, "module") else model

#         # 1) MMDet checkpoint: always use MMCV loader (handles spconv formats correctly)
#         if isinstance(ckpt, dict) and "state_dict" in ckpt:
#             print(f"[{self.name}] Using MMCV load_checkpoint for MMDet checkpoint: {load_path}", flush=True)
#             load_checkpoint(model_to_load, str(load_path), map_location="cpu", strict=False)
#             return

#         # 2) Raw state_dict (your FL weights): DO NOT call load_state_dict()
#         print(f"[{self.name}] Loading raw state_dict via MANUAL COPY: {load_path}", flush=True)
#         state_dict = ckpt

#         # Build name -> actual tensor reference maps
#         name_to_param = dict(model_to_load.named_parameters())
#         name_to_buf = dict(model_to_load.named_buffers())

#         loaded = 0
#         skipped = 0
#         shape_mismatch = 0

#         for k, v in state_dict.items():
#             target = None
#             if k in name_to_param:
#                 target = name_to_param[k]
#             elif k in name_to_buf:
#                 target = name_to_buf[k]

#             if target is None:
#                 skipped += 1
#                 continue

#             if target.shape != v.shape:
#                 shape_mismatch += 1
#                 # print first few mismatches only
#                 if shape_mismatch <= 5:
#                     print(f"[{self.name}] SHAPE MISMATCH {k}: ckpt={tuple(v.shape)} model={tuple(target.shape)}", flush=True)
#                 continue

#             # Copy weights WITHOUT triggering module-specific load logic
#             target.data.copy_(v)
#             loaded += 1

#         print(f"[{self.name}] Manual load complete. loaded={loaded} skipped_missing={skipped} skipped_shape={shape_mismatch}", flush=True)

#     # ------------------------------------------------------------------
#     # SAVE
#     # ------------------------------------------------------------------
#     def _save_model_weights(self, model, save_path: Union[str, Path]):
#         save_path = Path(save_path)
#         save_path.parent.mkdir(parents=True, exist_ok=True)
#         model_to_save = model.module if hasattr(model, "module") else model
#         torch.save(model_to_save.state_dict(), str(save_path))

#     # ------------------------------------------------------------------
#     # TRAIN
#     # ------------------------------------------------------------------
#     def train(self, base_cfg, load_path, work_root: Path, lr: float):
#         cfg = self._build_client_cfg(
#             scenes=self.scenes,
#             base_cfg=base_cfg,
#             num_epochs=self.num_epochs,
#             token_to_name_path=self.token_to_name_path,
#             seed=self.seed,
#             lr=lr
#         )

#         meta = self._init_runner_meta(cfg)

#         print(f"[CLIENT {self.name}] START TRAINING", flush=True)

#         cfg.work_dir = str(work_root)
#         save_path = work_root / "weights.pth"

#         dataset = build_dataset(cfg.data.train)
#         num_samples = len(dataset)

#         model = build_model(
#             cfg.model,
#             train_cfg=cfg.get("train_cfg"),
#             test_cfg=cfg.get("test_cfg"),
#         )
#         model.init_weights()
#         model.CLASSES = dataset.CLASSES

#         # 🔥 THIS IS THE CRITICAL CALL
#         self._load_model_weights(model, load_path)

#         train_detector(
#             model,
#             [dataset],
#             cfg,
#             distributed=False,
#             validate=False,
#             meta=meta
#         )

#         self._save_model_weights(model, save_path)

#         print(f"[CLIENT {self.name}] FINISHED — saved to {save_path}", flush=True)
#         return save_path, num_samples
