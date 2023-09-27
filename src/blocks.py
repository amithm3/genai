import torch
from torch import nn


class ConvBlock(nn.Sequential):
    NORMS = [None, nn.BatchNorm2d, nn.InstanceNorm2d]

    def __init__(
            self,
            inp_channels: int,
            out_channels: int,
            activation: "nn.Module" = None,
            **kwargs,
    ):
        p = kwargs.pop("p", 0)
        n = kwargs.pop("n", 0)
        norm = kwargs.pop("norm", 0)
        down = kwargs.pop("down", True)
        CONV = nn.Conv2d if down else nn.ConvTranspose2d

        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.n = n

        layers = [
            CONV(inp_channels, out_channels, bias=not norm, padding_mode="reflect" if down else "zeros", **kwargs),
            self.NORMS[norm](out_channels) if norm else None,
            *[module
              for module in (
                  CONV(out_channels, out_channels, bias=not norm, padding_mode="reflect" if down else "zeros",
                       kernel_size=3, stride=1, padding=1),
                  self.NORMS[norm](out_channels) if norm else None
              )
              for _ in range(n)],
            activation,
            nn.Dropout(p) if p else None
        ]

        super().__init__(*[layer for layer in layers if layer is not None])


class ResidualConvBlock(ConvBlock):
    def __init__(self, *args, **kwargs):
        identity = kwargs.pop("identity", False)
        super().__init__(*args, **kwargs)
        self.shortcut = ConvBlock(self.inp_channels, self.out_channels,
                                  norm=kwargs.get("norm", 0), down=kwargs.get("down", True),
                                  kernel_size=1, stride=self[0].stride) if not identity else nn.Identity()

    def forward(self, x):
        return super().forward(x) + self.shortcut(x)


class SkipConvBlock(nn.Module):
    def __init__(self, encoder: "nn.ModuleList", blocks: "nn.Module", decoder: "nn.ModuleList"):
        super().__init__()
        self.encoder = encoder
        self.blocks = blocks
        self.decoder = decoder

    def forward(self, x):
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
        x = self.blocks(x)
        for block in self.decoder:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block(x)
        return x
