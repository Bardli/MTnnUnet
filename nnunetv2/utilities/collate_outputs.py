from typing import List

import numpy as np
import torch


def collate_outputs(outputs: List[dict]):
    """
    used to collate default train_step and validation_step outputs. If you want something different then you gotta
    extend this

    we expect outputs to be a list of dictionaries where each of the dict has the same set of keys
    """
    collated = {}
    for k in outputs[0].keys():
        v0 = outputs[0][k]
        if np.isscalar(v0):
            collated[k] = [o[k] for o in outputs]
        elif isinstance(v0, np.ndarray):
            collated[k] = np.vstack([o[k][None] for o in outputs])
        elif torch.is_tensor(v0):
            try:
                collated[k] = torch.stack([o[k] for o in outputs])
            except Exception:
                # fallback to list if shapes mismatch
                collated[k] = [o[k] for o in outputs]
        elif isinstance(v0, list):
            collated[k] = [item for o in outputs for item in o[k]]
        else:
            raise ValueError(f'Cannot collate input of type {type(v0)}. '
                             f'Modify collate_outputs to add this functionality')
    return collated
