from dataclasses import dataclass
from typing import Iterable, TYPE_CHECKING

import torch
from torch import nn

from blocks import LinearBlock, ConvBlock, ResidualConvBlock, SkipBlock
from .gan import GanConfig, Gan
from .generator import Generator
from .discriminator import Discriminator

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class DCGanGenerator(Generator):
    def __init__(self, inp_features: int, out_channels: int, hidden_channels: int, **kwargs):
        n = kwargs.pop("n", 1)
        p = kwargs.pop("p", 0)
        norm = kwargs.pop("norm", nn.InstanceNorm2d)
        act = kwargs.pop("act", nn.ReLU())
        upsample = kwargs.pop("upsample", 2)
        features = kwargs.pop("features", 4)
        residuals = kwargs.pop("residuals", 9)
        assert kwargs == {}, \
            f"Unused arguments: {kwargs}"
        super().__init__(inp_features, out_channels, hidden_channels, n=n, p=p, norm=norm, act=act,
                         upsample=upsample, residuals=residuals, features=features)

    @staticmethod
    def build_head(inp_features: int, hidden_channels: int, **kwargs) -> "nn.Module":
        n = kwargs.pop("n")
        p = kwargs.pop("p")
        act = kwargs.pop("act")
        upsample = kwargs.pop("upsample")
        features = kwargs.pop("features")

        return nn.Sequential(
            LinearBlock(inp_features, hidden_channels * features * features, act,
                        n=n, p=p, act_every_n=False, norm_every_n=True),
            nn.Unflatten(1, (hidden_channels, features, features))
        )

    @staticmethod
    def build_blocks(hidden_channels: int, **kwargs) -> "nn.Module":
        n = kwargs.pop("n")
        p = kwargs.pop("p")
        norm = kwargs.pop("norm")
        act = kwargs.pop("act")
        upsample = kwargs.pop("upsample")
        residuals = kwargs.pop("residuals")

        residual_blocks = nn.Sequential(*[
            ResidualConvBlock(hidden_channels, hidden_channels, act, norm,
                              identity=True,
                              n=n, p=p, act_every_n=False, norm_every_n=True,
                              kernel_size=3, stride=1, padding=1)
            for _ in range(residuals)
        ])
        upsample_blocks = nn.Sequential(*([
            ConvBlock(hidden_channels // 2 ** i, hidden_channels // 2 ** (i + 1), act, norm,
                      n=n, p=p, act_every_n=False, norm_every_n=True, down=False,
                      kernel_size=4, stride=2, padding=1)
            for i in range(upsample)
        ]))

        return nn.Sequential(residual_blocks, upsample_blocks)

    @staticmethod
    def build_pred(hidden_channels: int, out_channels: int, **kwargs) -> "nn.Module":
        from blocks import ConvBlock
        n = kwargs.pop("n")
        p = kwargs.pop("p")
        upsample = kwargs.pop("upsample")

        return ConvBlock(hidden_channels // 2 ** upsample, out_channels, nn.Tanh(),
                         n=n, p=p, act_every_n=False, norm_every_n=False,
                         kernel_size=7, stride=1, padding=3)


class ConvDiscriminator(Discriminator):
    def __init__(self, inp_channels: int, blocks: Iterable[int], **kwargs):
        n = kwargs.pop("n", 1)
        p = kwargs.pop("p", 0)
        norm = kwargs.pop("norm", nn.InstanceNorm2d)
        act = kwargs.pop("act", nn.LeakyReLU(0.2))
        assert kwargs == {}, \
            f"Unused arguments: {kwargs}"
        blocks = list(blocks)
        super().__init__(inp_channels, blocks, n=n, p=p, norm=norm, act=act)

    @staticmethod
    def build_head(inp_channels: int, first_channels: int, **kwargs) -> "nn.Module":
        from blocks import ConvBlock
        n = kwargs.pop("n")
        p = kwargs.pop("p")
        act = kwargs.pop("act")

        return ConvBlock(inp_channels, first_channels, act,
                         n=n, p=p, act_every_n=False, norm_every_n=True,
                         kernel_size=4, stride=2, padding=1)

    @staticmethod
    def build_blocks(blocks: list[int], **kwargs) -> "nn.Module":
        from blocks import ConvBlock
        n = kwargs.pop("n")
        p = kwargs.pop("p")
        norm = kwargs.pop("norm")
        act = kwargs.pop("act")

        blocks = nn.Sequential(*[
            ConvBlock(inp_features, out_features, act, norm,
                      n=n, p=p, act_every_n=False, norm_every_n=True,
                      kernel_size=4, stride=2 if i < len(blocks) - 2 else 1, padding=1)
            for i, (inp_features, out_features) in enumerate(zip(blocks[:-1], blocks[1:]))
        ])

        return blocks

    @staticmethod
    def build_pred(final_channels: int, **kwargs) -> "nn.Module":
        from blocks import ConvBlock
        n = kwargs.pop("n", 1)
        p = kwargs.pop("p", 0)

        return ConvBlock(final_channels, 1, nn.Sigmoid(),
                         n=n, p=p, act_every_n=False, norm_every_n=False,
                         kernel_size=4, stride=1, padding=1)


@dataclass
class DCGanConfig(GanConfig):
    pass


class DCGan(Gan):
    def __init__(self, config: "DCGanConfig"):
        super().__init__(
            DCGanGenerator(config.inp_features, config.out_features, config.latent_features,
                           n=config.n, p=config.p, norm=config.norm, act=nn.ReLU(),
                           upsample=config.downsample, features=8, residuals=config.residuals),
            ConvDiscriminator(config.out_features, config.blocks,
                              n=config.n, p=config.p, norm=config.norm, act=nn.LeakyReLU(0.2))
        )
