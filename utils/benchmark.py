from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from tqdm import tqdm as TQDM

BAR_FORMAT = "{desc} {n_fmt}/{total_fmt}|{bar}|{percentage:3.0f}% [{elapsed}<{remaining}, {rate_fmt} {postfix}]"


class PerceptualLoss:
    def __init__(self, model: "nn.Module", criterion=None):
        if criterion is None: criterion = nn.L1Loss()
        self.model = model.eval().requires_grad_(False)
        self.criterion = criterion

    def __call__(self, x, y):
        return self.criterion(self.model(x), self.model(y))


def train(
        train_step: Callable[[dict[str, "torch.Tensor"], int], dict[str, float]],
        ds: "Dataset",
        ne: int = 10,
        bs: int = 32,
        collate_fn: Callable[[list[dict[str, "torch.Tensor"]]], dict[str, "torch.Tensor"]] = None,
        step_offset: int = 0,
):
    dl = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    step = step_offset
    for epoch in range(ne):
        metrics_sum = {}
        prog: "TQDM" = tqdm(dl, desc=f"Epoch: 0/{ne} | Batch", postfix={"loss": "?"}, bar_format=BAR_FORMAT)
        for batch, DATA in enumerate(prog):
            metrics = train_step(DATA, step)
            for k, v in metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0) + v
            prog.set_description(f"Epoch: {epoch + 1}/{ne} | Batch")
            prog.set_postfix(**{k: f"{v / (batch + 1):.4f}" for k, v in metrics_sum.items()})
            step += 1
    return step


def test(
        tester: Callable[[dict[str, "torch.Tensor"]], dict[str, float]],
        ds: "Dataset",
        bs: int = 32,
        collate_fn: Callable[[list[dict[str, "torch.Tensor"]]], dict[str, "torch.Tensor"]] = None
):
    dl = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    metrics_sum = {}
    prog: "TQDM" = tqdm(dl, desc=f"Batch", postfix={"loss": "?"}, bar_format=BAR_FORMAT)
    for batch, DATA in enumerate(prog):
        metrics = tester(DATA)
        for k, v in metrics.items():
            metrics_sum[k] = metrics_sum.get(k, 0) + v
        prog.set_postfix(**{k: f"{v / (batch + 1):.4f}" for k, v in metrics_sum.items()})


__all__ = [
    "PerceptualLoss",
    "train",
    "test",
]
