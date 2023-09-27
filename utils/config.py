import os
from dataclasses import dataclass
from typing import Union, TYPE_CHECKING

import torch
import torchvision.transforms as T
from torch import nn

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


@dataclass
class Config:
    dataset_path: str
    model_name: str
    model_version: str = 'v1'
    model_dir: str = "./models/"
    log_dir: str = "./logs/"
    device: str = "cuda" if torch.has_cuda else "mps" if torch.has_mps else "cpu"
    image_shape: tuple = 3, 224, 224

    num_epochs: int = 10
    batch_size: int = 32
    lr: float = 2e-4
    betas: tuple[float, ...] = ()
    alphas: tuple[float, ...] = ()
    lambdas: tuple[float, ...] = ()
    dropout: float = 0

    writer: Union["SummaryWriter", bool] = False
    mean: tuple[float, ...] = None
    std: tuple[float, ...] = None
    transforms: "T.Compose" = None
    denorm: "T.Normalize" = None

    @property
    def checkpoint_path(self) -> str:
        return f"{self.model_dir}/{self.model_name}/{self.model_version}/"

    @property
    def log_path(self) -> str:
        return f"{self.log_dir}/{self.model_name}/{self.model_version}/"

    def __post_init__(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        if self.writer:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_path)
        if not self.mean: self.mean = (0.5,) * self.image_shape[0]
        if not self.std: self.std = (0.5,) * self.image_shape[0]
        self.transforms = T.Compose([
            T.Resize(self.image_shape[1:]),
            T.ToTensor(),
            T.Normalize(self.mean, self.std),
            lambda x: x.to(self.device),
        ])
        self.denorm = T.Normalize(-torch.tensor(self.mean) / torch.tensor(self.std), 1 / torch.tensor(self.std))

    def copy(self, **kwargs):
        return type(self)(**{**self.__dict__, **kwargs})


def weights_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0.0)


__all__ = [
    "Config",
    "weights_init",
]
