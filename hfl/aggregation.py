from pathlib import Path
from typing import Union
import torch

from mmcv import print_log

def average_weights(weight_paths, sample_counts):
    """ Compute weighted average of model parameters

    Args:
        weight_paths (List[str|Path]): List of file paths where model weights are stored
            as .pth files storing state dict or cpkts.
        sample_counts (List[int]): List of integers describing across how many samples
            each model weight is trained for weighted averaging.
    """
    assert len(weight_paths) == len(sample_counts), \
        f"Lists containing file paths to weights (.pth files) and " \
        f"corresponding training samples are of inconsistent length."

    total_samples = sum(sample_counts)
    assert total_samples > 0, \
        f"Cannot average over a non-positive number of training samples."

    avg_weights = None
    for path, samples in zip(weight_paths, sample_counts):
        # Load state dict or ckpts
        ckpt = torch.load(str(path), map_location="cpu")
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        w = samples / total_samples

        if avg_weights is None:
            avg_weights = {k: v * w for k, v in state_dict.items()}
        else:
            for k in avg_weights:
                avg_weights[k] += state_dict[k] * w

    return avg_weights

def save_state_dict(state_dict: dict, weights_path: Union[str, Path]):
    """ Store a state_dict.
    
    Args:
        state_dict (Dict): State dict containing model weight assignment.
        weights_path (str|Path): File path where to store state dict.
    """
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, str(weights_path))
