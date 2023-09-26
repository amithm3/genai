from datetime import datetime

import torch
from torch import nn, optim


def save_checkpoint(
        path: str,
        models: dict[str, "nn.Module"],
        optimizers: dict[str, "optim.Optimizer"],
        **others,
) -> str:
    path = f"{path}/checkpoint-{datetime.now()}.pt" if not path.endswith(".pt") else path
    torch.save({
        **{f"model_{k}_state_dict": v.state_dict() for k, v in models.items()},
        **{f"optim_{k}_state_dict": v.state_dict() for k, v in optimizers.items()},
        **others
    }, path)
    return path


def load_checkpoint(
        path: str,
        models: dict[str, "nn.Module"],
        optimizers: dict[str, "optim.Optimizer"] = None,
        device: str = "cpu"
) -> dict:
    checkpoint = torch.load(path, map_location=device)
    for k, v in models.items():
        v.load_state_dict(checkpoint.pop(f"model_{k}_state_dict"))
    if optimizers is not None:
        for k, v in optimizers.items():
            v.load_state_dict(checkpoint.pop(f"optim_{k}_state_dict"))
    return checkpoint


__all__ = [
    "save_checkpoint",
    "load_checkpoint",
]
