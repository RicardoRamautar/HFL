from hfl.lr_scheduler import LRScheduler

from typing import List, Optional, Union
from pathlib import Path
import copy

from mmcv.utils import import_modules_from_strings
from mmcv.runner import load_checkpoint
from mmcv import print_log
from mmcv import Config
import torch

from mmdet3d.apis.train import train_detector
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


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
                 lr_cfg: dict,
                 num_epochs: int = 1,
                 token_to_name_path: Optional[str] = None,
                 seed: int = 0, 
                 resume_from: int = 0):

        if num_epochs <= 0:
            raise ValueError(f"Number of training epochs must be a positive integer.")

        self.name = name
        self.scenes = scenes
        # self.lr_scheduler = LRScheduler(**lr_cfg)
        self.lr_cfg = copy.deepcopy(lr_cfg)
        self.num_epochs = num_epochs
        self.token_to_name_path = token_to_name_path
        self.seed = seed
        self.num_samples = 0
        self.total_rounds = resume_from

        self.lr_cfg['offset'] = self.total_rounds


    def _build_client_cfg(self, 
                          scenes: List[str], 
                          base_cfg: Config, 
                          num_epochs: int, 
                          token_to_name_path: Union[str, None], 
                          seed: int,
                          ):
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

        # cfg.data.train.enable_hfl = True
        cfg.data.train.dataset.enable_hfl = True

        # cfg.data.train.scenes = scenes
        cfg.data.train.dataset.scenes = scenes

        if token_to_name_path is not None:
            # cfg.data.train.scene_translation = token_to_name_path
            cfg.data.train.dataset.scene_translation = token_to_name_path
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

        cfg.custom_hooks = [
            {
                'type':'CyclicLrPerIter',
                'lr_cfg': copy.deepcopy(self.lr_cfg)
            },
            {
                'type': 'StoreHFLMetrics'
            }
        ]

        return cfg


    def _init_runner_meta(self, cfg):
        """Minimal initialization of runner.meta similar to tools/train.py."""
        meta = {}

        # Seed handling (required)
        meta["seed"] = cfg.seed

        # Optional but useful
        meta["exp_name"] = "hfl_client"
        meta["config"] = cfg.pretty_text

        meta["train_metrics"] = []

        return meta


    def _load_model_weights(self, model, load_path: Union[str, Path]):
        load_path = Path(load_path)
        ckpt = torch.load(str(load_path), map_location="cpu")

        model_to_load = model.module if hasattr(model, "module") else model

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            print(f"[{self.name}] Using MMCV load_checkpoint for MMDet checkpoint: {load_path}", flush=True)
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
              work_root: Path):
        """ Train model on client dataset.

        Args:
            load_path (str | Path): File path containing initial weights.
            root (Path): Folder path where the training results will be stored.
            save_path (Path): File path where updated weights will be stored.
            round_idx (int): How many edge aggregations rounds have been performed.
        """
        print_log(f"[CLIENT - {self.name}] has started training")

        cfg = self._build_client_cfg(
            scenes = self.scenes,
            base_cfg = base_cfg,
            num_epochs = self.num_epochs,
            token_to_name_path = self.token_to_name_path,
            seed = self.seed,
        )
        meta = self._init_runner_meta(cfg)

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
        if load_path is not None:
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

        self.total_rounds += self.num_epochs
        # self.lr_scheduler.set_offset(self.total_rounds)
        self.lr_cfg['offset'] = self.total_rounds

        # Store weights
        self._save_model_weights(model, save_path)

        training_results = {"id": self.name, "num_samples": num_samples, "training_results": meta['train_metrics']}
        
        print(f"Client {self.name} stored weights at {str(save_path)}")
        return save_path, num_samples, training_results

