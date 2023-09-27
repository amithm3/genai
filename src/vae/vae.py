from abc import abstractmethod, ABCMeta
import math

import torch
from torch import nn

from blocks import ConvBlock
from lambdas import Parallel


class VAE(nn.Module, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def build_encoder(inp_features: int, latent_features: int, blocks, **kwargs) -> "nn.Module":
        pass

    @staticmethod
    @abstractmethod
    def build_decoder(inp_features: int, latent_features: int, blocks, **kwargs) -> "nn.Module":
        pass

    @staticmethod
    def reparameterize(mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
        if torch.is_grad_enabled():
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def __init__(self, inp_features: int, latent_features: int, blocks, **kwargs):
        super().__init__()
        self.inp_features = inp_features
        self.latent_features = latent_features
        self.blocks = blocks

        self.encoder = self.build_encoder(inp_features, latent_features, blocks[::+1], **kwargs)
        self.decoder = self.build_decoder(inp_features, latent_features, blocks[::-1], **kwargs)

    def encode(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar), mu, logvar

    def decode(self, z: "torch.Tensor") -> "torch.Tensor":
        return self.decoder(z)

    def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        z, mu, logvar = self.encode(x)
        return self.decode(z), mu, logvar


class ConvVAE(VAE):
    @staticmethod
    def build_encoder(inp_features: int, latent_features: int, blocks, **kwargs) -> "nn.Module":
        n = kwargs.get('n', 1)
        p = kwargs.get('p', 0)
        norm = kwargs.get('norm', 1)

        return nn.Sequential(
            ConvBlock(inp_features, blocks[0], nn.LeakyReLU(0.2),
                      n=n, p=p, norm=0, kernel_size=7, stride=1, padding=3),
            *[
                ConvBlock(inp, out, nn.LeakyReLU(0.2),
                          n=n, p=p, norm=norm, kernel_size=4, stride=2, padding=1)
                for inp, out in zip(blocks[:-1], blocks[1:])
            ],
            nn.Identity(),
            Parallel(
                nn.Identity(),
                nn.Identity(),
            ),
        )

    @staticmethod
    def build_decoder(inp_features: int, latent_features: int, blocks, **kwargs) -> "nn.Module":
        n = kwargs.get('n', 1)
        p = kwargs.get('p', 0)
        norm = kwargs.get('norm', 1)

        return nn.Sequential(
            nn.Identity(),
            nn.Identity(),
            *[
                ConvBlock(inp, out, nn.ReLU(), down=False,
                          n=n, p=p, norm=norm, kernel_size=4, stride=2, padding=1)
                for inp, out in zip(blocks[:-1], blocks[1:])
            ],
            ConvBlock(blocks[-1], inp_features, nn.Tanh(), down=False,
                      n=n, p=p, norm=0, kernel_size=7, stride=1, padding=3),
        )

    def _encode(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        with torch.no_grad():
            y1, y2 = self.encoder(x)

        self.encoder[-2] = nn.Flatten()
        flat = math.prod(y1.shape[1:])
        self.encoder[-1][0] = nn.Linear(flat, self.latent_features)
        self.encoder[-1][1] = nn.Linear(flat, self.latent_features)

        self.decoder[0] = nn.Linear(self.latent_features, flat)
        self.decoder[1] = nn.Unflatten(1, y1.shape[1:])

        self.encode = super().encode
        return super().encode(x)

    encode = _encode


def test_ConvVAE():
    vae = ConvVAE(3, 32, (16, 32, 48, 64, 80, 96))
    x = torch.randn(4, 3, 224, 224)
    y = vae(x)
    print(vae)
    print(x.shape, y[0].shape, y[1].shape, y[2].shape)


if __name__ == '__main__':
    test_ConvVAE()
