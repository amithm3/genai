from abc import ABCMeta, abstractmethod

from torch import nn


class Generator(nn.Module, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def build_head(inp: int, hidden: int, **kwargs) -> "nn.Module":
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build_blocks(hidden: int, **kwargs) -> "nn.Module":
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build_pred(hidden: int, out: int, **kwargs) -> "nn.Module":
        raise NotImplementedError

    def __init__(self, inp: int, out: int, hidden: int, **kwargs):
        super().__init__()

        self.head = self.build_head(inp, hidden, **kwargs)
        self.blocks = self.build_blocks(hidden, **kwargs)
        self.pred = self.build_pred(hidden, out, **kwargs)

    def forward(self, x):
        x = self.head(x)
        x = self.blocks(x)
        y = self.pred(x)
        return y


class LinearGenerator(Generator):
    def __init__(self, inp_features: int, out_features: int, latent_features: int, **kwargs):
        n = kwargs.pop("n", 1)
        p = kwargs.pop("p", 0)
        norm = kwargs.pop("norm", nn.InstanceNorm1d)
        act = kwargs.pop("act", nn.ReLU())
        downsample = kwargs.pop("downsample", 2)
        residuals = kwargs.pop("residuals", 9)
        assert kwargs == {}, \
            f"Unused arguments: {kwargs}"
        super().__init__(inp_features, out_features, latent_features, n=n, p=p, norm=norm, act=act,
                         downsample=downsample, residuals=residuals)

    @staticmethod
    def build_head(inp_features: int, latent_features: int, **kwargs) -> "nn.Module":
        from blocks import LinearBlock
        n = kwargs.pop("n")
        p = kwargs.pop("p")
        act = kwargs.pop("act")

        return LinearBlock(inp_features, latent_features, act,
                           n=n, p=p, act_every_n=False, norm_every_n=True)

    @staticmethod
    def build_blocks(latent_features: int, **kwargs) -> "nn.Module":
        from blocks import LinearBlock, ResidualLinearBlock, SkipBlock
        n = kwargs.pop("n")
        p = kwargs.pop("p")
        norm = kwargs.pop("norm")
        act = kwargs.pop("act")
        downsample = kwargs.pop("downsample")
        residuals = kwargs.pop("residuals")

        encoder_blocks = nn.ModuleList([
            LinearBlock(latent_features // 2 ** i, latent_features // 2 ** (i + 1), act, norm,
                        n=n, p=p, act_every_n=False, norm_every_n=True)
            for i in range(downsample)
        ])
        residual_blocks = nn.Sequential(*[
            ResidualLinearBlock(latent_features // 2 ** downsample, latent_features // 2 ** downsample, act, norm,
                                identity=True,
                                n=n, p=p, act_every_n=False, norm_every_n=True)
            for _ in range(residuals)
        ])
        decoder_blocks = nn.ModuleList(reversed([
            LinearBlock(latent_features // 2 ** (i + 1) * 2, latent_features // 2 ** i, act, norm,
                        n=n, p=p, act_every_n=False, norm_every_n=True)
            for i in range(downsample)
        ]))

        return SkipBlock(encoder_blocks, residual_blocks, decoder_blocks)

    @staticmethod
    def build_pred(latent_features: int, out_features: int, **kwargs) -> "nn.Module":
        from blocks import LinearBlock
        n = kwargs.pop("n")
        p = kwargs.pop("p")

        return LinearBlock(latent_features, out_features, nn.Tanh(),
                           n=n, p=p, act_every_n=False, norm_every_n=False)


def test_LinearGenerator():
    import torch

    inp_features = 300
    out_features = 500
    latent_features = 300
    batch_size = 7
    x = torch.randn(batch_size, inp_features)
    model = LinearGenerator(inp_features, out_features, latent_features)
    print(model)
    y = model(x)
    assert y.shape == (batch_size, out_features)


if __name__ == '__main__':
    test_LinearGenerator()

__all__ = [
    "Generator",
]
