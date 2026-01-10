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
                #  base_cfg: Config,
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

        # self.cfg = self._build_client_cfg(
        #     scenes = scenes,
        #     base_cfg = base_cfg,
        #     num_epochs = num_epochs,
        #     token_to_name_path = token_to_name_path,
        #     seed = seed       
        # )
        # self.meta = self._init_runner_meta()



    def _build_client_cfg(self, 
                          scenes: List[str], 
                          base_cfg: Config, 
                          num_epochs: int, 
                          token_to_name_path: Union[str, None], 
                          seed: int):
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

        cfg.gpu_ids = [0]

        # We set the model weights ourselves
        cfg.load_from = None
        cfg.resume_from = None

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


    def _load_model_weights(self, model, load_path: Union[str, Path]):
        """ Load model weights from file."""
        ckpt = torch.load(str(load_path), map_location="cpu")

        # Support model-only weights and MMDET cktps
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        model_to_load = model.module if hasattr(model, "module") else model
        model_to_load.load_state_dict(state_dict, strict=False)


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
            load_path (str | Path): :File path containing initial weights.
            root (Path): Folder path where the training results will be stored.
            save_path (Path): File path where updated weights will be stored.
        """
        cfg = self._build_client_cfg(
            scenes = self.scenes,
            base_cfg = base_cfg,
            num_epochs = self.num_epochs,
            token_to_name_path = self.token_to_name_path,
            seed = self.seed       
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
