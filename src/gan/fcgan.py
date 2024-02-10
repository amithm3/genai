from dataclasses import dataclass
from typing import Iterable, TYPE_CHECKING

import torch
from torch import nn

from .gan import GanConfig, Gan
from .generator import Generator
from .discriminator import Discriminator

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class ConvGenerator(Generator):
    def __init__(self, inp_channels: int, out_channels: int, latent_channels: int, **kwargs):
        n = kwargs.pop("n", 1)
        p = kwargs.pop("p", 0)
        norm = kwargs.pop("norm", nn.InstanceNorm2d)
        act = kwargs.pop("act", nn.ReLU())
        downsample = kwargs.pop("downsample", 2)
        residuals = kwargs.pop("residuals", 9)
        assert kwargs == {}, \
            f"Unused arguments: {kwargs}"
        super().__init__(inp_channels, out_channels, latent_channels, n=n, p=p, norm=norm, act=act,
                         downsample=downsample, residuals=residuals)

    @staticmethod
    def build_head(inp_channels: int, latent_channels: int, **kwargs) -> "nn.Module":
        from blocks import ConvBlock
        n = kwargs.pop("n")
        p = kwargs.pop("p")
        act = kwargs.pop("act")

        return ConvBlock(inp_channels, latent_channels, act,
                         n=n, p=p, act_every_n=False, norm_every_n=True,
                         kernel_size=7, stride=1, padding=3)

    @staticmethod
    def build_blocks(latent_channels: int, **kwargs) -> "nn.Module":
        from blocks import ConvBlock, ResidualConvBlock, SkipBlock
        n = kwargs.pop("n")
        p = kwargs.pop("p")
        norm = kwargs.pop("norm")
        act = kwargs.pop("act")
        downsample = kwargs.pop("downsample")
        residuals = kwargs.pop("residuals")

        encoder_blocks = nn.ModuleList([
            ConvBlock(latent_channels * 2 ** i, latent_channels * 2 ** (i + 1), act, norm,
                      n=n, p=p, act_every_n=False, norm_every_n=True, down=True,
                      kernel_size=4, stride=2, padding=1)
            for i in range(downsample)
        ])
        residual_blocks = nn.Sequential(*[
            ResidualConvBlock(latent_channels * 2 ** downsample, latent_channels * 2 ** downsample, act, norm,
                              identity=True,
                              n=n, p=p, act_every_n=False, norm_every_n=True,
                              kernel_size=3, stride=1, padding=1)
            for _ in range(residuals)
        ])
        decoder_blocks = nn.ModuleList(reversed([
            ConvBlock(latent_channels * 2 ** (i + 1) * 2, latent_channels * 2 ** i, act, norm,
                      n=n, p=p, act_every_n=False, norm_every_n=True, down=False,
                      kernel_size=4, stride=2, padding=1)
            for i in range(downsample)
        ]))

        return SkipBlock(encoder_blocks, residual_blocks, decoder_blocks)

    @staticmethod
    def build_pred(latent_channels: int, out_channels: int, **kwargs) -> "nn.Module":
        from blocks import ConvBlock
        n = kwargs.pop("n")
        p = kwargs.pop("p")

        return ConvBlock(latent_channels, out_channels, nn.Tanh(),
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
class FCGanConfig(GanConfig):
    pass


class FCGan(Gan):
    def __init__(self, config: "FCGanConfig"):
        super().__init__(
            ConvGenerator(config.inp_features, config.out_features, config.latent_features,
                          n=config.n, p=config.p, norm=config.norm, act=nn.ReLU(),
                          downsample=config.downsample, residuals=config.residuals),
            ConvDiscriminator(config.out_features, config.blocks,
                              n=config.n, p=config.p, norm=config.norm, act=nn.LeakyReLU(0.2))
        )
