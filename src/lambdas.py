from torch import nn


class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class Parallel(nn.ModuleList):
    def __init__(self, *modules: "nn.Module"):
        super().__init__(modules)

    def forward(self, x):
        return [module(x) for module in self]


__all__ = [
    "Lambda",
    "Parallel",
]
