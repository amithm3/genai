from abc import ABCMeta, abstractmethod

from torch import nn

from blocks import SkipConvBlock


class Generator(nn.Module, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def build_head(inp_features: int, latent_features: int, **kwargs) -> "nn.Module":
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build_blocks(latent_features: int, **kwargs) -> tuple["nn.ModuleList", "nn.Module", "nn.ModuleList"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build_pred(latent_features: int, out_features: int, **kwargs) -> "nn.Module":
        raise NotImplementedError

    def __init__(self, inp_features: int, out_features: int, latent_features: int, **kwargs):
        super().__init__()

        self.head = self.build_head(inp_features, latent_features, **kwargs)
        self.blocks = SkipConvBlock(*self.build_blocks(latent_features, **kwargs))
        self.pred = self.build_pred(latent_features, out_features, **kwargs)

    def forward(self, x):
        x = self.head(x)
        x = self.blocks(x)
        return self.pred(x)


class ConvGenerator(Generator):
    @staticmethod
    def build_head(inp_features: int, latent_features: int, **kwargs) -> "nn.Module":
        from blocks import ConvBlock

        n = kwargs.pop("n", 1)
        p = kwargs.pop("p", 0)

        return ConvBlock(inp_features, latent_features, nn.ReLU(),
                         norm=0, kernel_size=7, stride=1, padding=3, n=n, p=p)

    @staticmethod
    def build_blocks(latent_features: int, **kwargs) -> tuple["nn.ModuleList", "nn.Module", "nn.ModuleList"]:
        from blocks import ConvBlock, ResidualConvBlock

        n = kwargs.pop("n", 1)
        p = kwargs.pop("p", 0)
        norm = kwargs.pop("norm", 2)
        downsample = kwargs.pop("downsample", 2)
        residuals = kwargs.pop("residuals", 9)

        encoder_blocks = nn.ModuleList([
            ConvBlock(latent_features * 2 ** i, latent_features * 2 ** (i + 1), nn.ReLU(),
                      norm=norm, kernel_size=3, stride=2, padding=1, n=n, p=p)
            for i in range(downsample)
        ])
        residual_blocks = nn.Sequential(*[
            ResidualConvBlock(latent_features * 2 ** downsample, latent_features * 2 ** downsample,
                              identity=True, norm=norm, kernel_size=3, stride=1, padding=1, n=n, p=p)
            for _ in range(residuals)
        ])
        decoder_blocks = nn.ModuleList(reversed([
            ConvBlock(latent_features * 2 ** (i + 1) * 2, latent_features * 2 ** i, nn.ReLU(),
                        norm=norm, down=False, kernel_size=3, stride=2, padding=1, output_padding=1, n=n, p=p)
            for i in range(downsample)
        ]))

        return encoder_blocks, residual_blocks, decoder_blocks

    @staticmethod
    def build_pred(latent_features: int, out_features: int, **kwargs) -> "nn.Module":
        from blocks import ConvBlock
        n = kwargs.pop("n", 1)
        p = kwargs.pop("p", 0)
        return ConvBlock(latent_features, out_features, nn.Tanh(),
                         norm=0, kernel_size=7, stride=1, padding=3, n=n, p=p)


def test_ConvGenerator():
    import torch

    latent_features = 32
    inp_features = 3
    out_features = 3
    batch_size = 4
    image_size = 128
    x = torch.randn(batch_size, inp_features, image_size, image_size)
    model = ConvGenerator(inp_features, out_features, latent_features, downsample=3, residuals=9)
    print(model)
    print(model(x).shape)


if __name__ == '__main__':
    test_ConvGenerator()


__all__ = [
    "Generator",
    "ConvGenerator"
]
