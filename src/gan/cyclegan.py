from .config import CycleGANConfig
from .generator import ConvGenerator
from .discriminator import ConvDiscriminator


def get_conv_generators(
        config: "CycleGANConfig",
):
    return (
        ConvGenerator(
            config.inp_features,
            config.out_features,
            config.latent_features,
            n=config.n,
            p=config.p,
            norm=config.norm,
            downsample=config.downsample,
            residuals=config.residuals,
        ),
        ConvGenerator(
            config.inp_features,
            config.out_features,
            config.latent_features,
            n=config.n,
            p=config.p,
            norm=config.norm,
            downsample=config.downsample,
            residuals=config.residuals,
        ),
    )


def get_conv_discriminators(
        config: "CycleGANConfig",
):
    return (
        ConvDiscriminator(
            config.inp_features,
            config.blocks,
            n=config.n,
            p=config.p,
            norm=config.norm,
        ),
        ConvDiscriminator(
            config.inp_features,
            config.blocks,
            n=config.n,
            p=config.p,
            norm=config.norm,
        ),
    )


__all__ = [
    "get_conv_generators",
    "get_conv_discriminators",
]
