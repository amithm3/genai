from dataclasses import dataclass

from utils.config import Config


@dataclass
class CycleGANConfig(Config):
    latent_dim: int = 64
    downsample: int = 2
    residuals: int = 9
    blocks: tuple = (64, 128, 256, 512)
    betas: tuple[float, float] = (0.5, 0.999)
    lambdas: tuple[float, float] = (10, 0.5)
    n: int = 0


__all__ = [
    "CycleGANConfig",
]
