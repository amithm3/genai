from dataclasses import dataclass

from utils.config import Config


@dataclass
class VAEConfig(Config):
    latent_dim: int = 64
    blocks: tuple = (64, 128, 256, 512)
    lambdas: tuple[float] = (10,)
    n: int = 0


__all__ = [
    "VAEConfig",
]
