from abc import ABCMeta, abstractmethod

from torch import nn


class Discriminator(nn.Module, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def get_head(inp_features: int, first_features: int, **kwargs) -> "nn.Module":
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_blocks(blocks: list[int], **kwargs) -> "nn.Module":
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_pred(inp_features: int, final_features: int, **kwargs) -> "nn.Module":
        raise NotImplementedError

    def __init__(self, inp_features: int, blocks: list[int], **kwargs):
        super().__init__()
        assert len(blocks) > 2

        self.head = self.get_head(inp_features, blocks[0], **kwargs)
        self.blocks = self.get_blocks(blocks, **kwargs)
        self.pred = self.get_pred(blocks[-2], blocks[-1], **kwargs)

    def forward(self, x):
        x = self.head(x)
        x = self.blocks(x)
        return self.pred(x)


class ConvDiscriminator(Discriminator):
    @staticmethod
    def get_head(inp_features: int, first_features: int, **kwargs) -> "nn.Module":
        from blocks import ConvBlock

        n = kwargs.pop("n", 1)
        p = kwargs.pop("p", 0)

        return ConvBlock(inp_features, first_features, nn.LeakyReLU(0.2),
                         norm=0, kernel_size=4, stride=2, padding=1, n=n, p=p)

    @staticmethod
    def get_blocks(blocks: list[int], **kwargs) -> "nn.Module":
        from blocks import ConvBlock

        n = kwargs.pop("n", 1)
        p = kwargs.pop("p", 0)
        norm = kwargs.pop("norm", 2)

        return nn.Sequential(*[
            ConvBlock(inp_features, out_features, nn.LeakyReLU(0.2),
                      norm=norm, kernel_size=4, stride=2, padding=1, n=n, p=p)
            for inp_features, out_features in zip(blocks[:-2], blocks[1:-1])
        ], ConvBlock(blocks[-2], blocks[-1], nn.LeakyReLU(0.2),
                     norm=norm, kernel_size=4, stride=1, padding=1, n=n, p=p))

    @staticmethod
    def get_pred(inp_features: int, final_features: int, **kwargs) -> "nn.Module":
        from blocks import ConvBlock

        n = kwargs.pop("n", 1)
        p = kwargs.pop("p", 0)

        return ConvBlock(final_features, 1, nn.Sigmoid(),
                         norm=0, kernel_size=4, stride=1, padding=1, n=n, p=p)


def test_discriminator():
    import torch

    img_shape = 5, 256, 256
    disc = ConvDiscriminator(img_shape[0], [64, 128, 256, 512])
    print(disc)
    print(disc(torch.rand(1, *img_shape)).shape)
    print(disc(torch.rand(7, *img_shape)).shape)


if __name__ == '__main__':
    test_discriminator()

__all__ = [
    "Discriminator",
    "ConvDiscriminator",
]
