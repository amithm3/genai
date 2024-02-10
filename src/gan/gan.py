from typing import TYPE_CHECKING
from dataclasses import dataclass

from torch import nn

from utils.config import Config

if TYPE_CHECKING:
    from .generator import Generator
    from .discriminator import Discriminator


@dataclass
class GanConfig(Config):
    inp_features: int = 3
    out_features: int = 3
    latent_features: int = 64
    downsample: int = 2
    residuals: int = 9
    n: int = 1
    blocks: tuple = (64, 128, 256, 512)
    betas: tuple = (0.5, 0.999)
    lambda_perceptual: float = 1


class Gan(nn.ModuleDict):
    @property
    def G(self) -> "nn.Module":
        return self["generator"]

    @property
    def D(self) -> "nn.Module":
        return self["discriminator"]

    def __init__(self, generator: "Generator", discriminator: "Discriminator"):
        super().__init__({
            "generator": generator,
            "discriminator": discriminator,
        })
