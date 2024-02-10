from abc import ABCMeta, abstractmethod
from typing import Iterable

from torch import nn


class Discriminator(nn.Module, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def build_head(inp: int, first: int, **kwargs) -> "nn.Module":
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build_blocks(blocks: list[int], **kwargs) -> "nn.Module":
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build_pred(final: int, **kwargs) -> "nn.Module":
        raise NotImplementedError

    def __init__(self, inp: int, blocks: Iterable[int], **kwargs):
        blocks = list(blocks)
        super().__init__()
        assert len(blocks) > 2

        self.head = self.build_head(inp, blocks[0], **kwargs)
        self.blocks = self.build_blocks(blocks, **kwargs)
        self.pred = self.build_pred(blocks[-1], **kwargs)

    def forward(self, x):
        x = self.head(x)
        x = self.blocks(x)
        y = self.pred(x)
        return y


class LinearDiscriminator(Discriminator):
    def __init__(self, inp_features: int, blocks: Iterable[int], **kwargs):
        n = kwargs.pop("n", 1)
        p = kwargs.pop("p", 0)
        norm = kwargs.pop("norm", nn.InstanceNorm1d)
        act = kwargs.pop("act", nn.ReLU())
        assert kwargs == {}, \
            f"Unused arguments: {kwargs}"
        super().__init__(inp_features, blocks, n=n, p=p, norm=norm, act=act)

    @staticmethod
    def build_head(inp_features: int, first_features: int, **kwargs) -> "nn.Module":
        from blocks import LinearBlock
        n = kwargs.pop("n")
        p = kwargs.pop("p")
        act = kwargs.pop("act")

        return LinearBlock(inp_features, first_features, act,
                           n=n, p=p, act_every_n=False, norm_every_n=True)

    @staticmethod
    def build_blocks(blocks: list[int], **kwargs) -> "nn.Module":
        from blocks import LinearBlock
        n = kwargs.pop("n")
        p = kwargs.pop("p")
        norm = kwargs.pop("norm")
        act = kwargs.pop("act")

        blocks = nn.Sequential(*[
            LinearBlock(inp_features, out_features, act, norm,
                        n=n, p=p, act_every_n=False, norm_every_n=True)
            for inp_features, out_features in zip(blocks[:-1], blocks[1:])
        ])

        return blocks

    @staticmethod
    def build_pred(final_features: int, **kwargs) -> "nn.Module":
        from blocks import LinearBlock
        p = kwargs.pop("p", 0)

        return LinearBlock(final_features, 1, nn.Sigmoid(),
                           n=0, p=p, act_every_n=False, norm_every_n=False)


def test_LinearDiscriminator():
    import torch

    inp_features = 500
    batch_size = 7
    x = torch.randn(batch_size, inp_features)
    disc = LinearDiscriminator(inp_features, [inp_features // 2, inp_features // 4, inp_features // 8, 16])
    print(disc)
    y = disc(x)
    print(y.shape)


if __name__ == '__main__':
    test_LinearDiscriminator()

__all__ = [
    "Discriminator",
]
