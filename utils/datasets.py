import os
import random
from pathlib import Path
from typing import Union, Generator
from abc import ABCMeta, abstractmethod

import torch
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset, metaclass=ABCMeta):
    SETS = None,

    @abstractmethod
    def get_data(self) -> Generator[dict[str, ...], None, None]:
        raise NotImplementedError

    @classmethod
    def download(cls, DIR: Union[str, "Path"]):
        raise NotImplementedError("This dataset does not support download")

    @staticmethod
    def kaggle_download(DIR: Union[str, "Path"], dataset: str):
        user = input("Enter your kaggle username: ")
        token = input("Enter your kaggle token: ")
        os.environ["KAGGLE_USERNAME"] = user
        os.environ["KAGGLE_KEY"] = token
        import kaggle
        kaggle.api.dataset_download_files(dataset, path=DIR, unzip=True, quiet=False)
        os.environ.pop("KAGGLE_USERNAME")
        os.environ.pop("KAGGLE_KEY")

    @property
    def DIR(self) -> "Path":
        return self._DIR

    @property
    def SET(self) -> str:
        return self._SET

    def __init__(
            self,
            DIR: Union[str, "Path"],
            SET: str = None,
            **kwargs
    ):
        download = kwargs.pop("download", False)
        sub_sample = kwargs.pop("sub_sample", 1)
        transforms = {k.removesuffix("_transform"): v for k, v in kwargs.items() if k.endswith("_transform")}
        if os.path.isdir(DIR) and len(os.listdir(DIR)): download = False
        if download:
            os.makedirs(DIR, exist_ok=True)
            self.download(DIR)
        assert os.path.isdir(DIR), \
            f"Directory {DIR} does not exist"
        assert 0 < sub_sample <= 1, \
            f"Value of sub_sample must be between 0 and 1, got {sub_sample}"
        assert SET in self.SETS, \
            f"invalid value of SET, must be one of {self.SETS}, got {SET}"

        self._DIR = Path(DIR)
        self._SET = SET
        self._T = transforms

        data = list(self.get_data())
        random.shuffle(data)
        self._data = data[:int(len(data) * sub_sample)]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item) -> dict[str, "torch.Tensor"]:
        try:
            item = iter(item)
            return self.collate_fn(self[idx] for idx in item)
        except TypeError:
            pass
        if isinstance(item, slice):
            return self.collate_fn(self[idx] for idx in range(*item.indices(len(self))))
        elif isinstance(item, int):
            data_item = dict(self._data[item % len(self)])
            file = data_item.pop("file")
            data_item["image"] = Image.open(file)
            return {
                k: self._T.get(k, lambda x: x)(v) for k, v in data_item.items()
            }
        else:
            raise TypeError(f"Invalid argument type {type(item)}")

    @staticmethod
    def collate_fn(batch) -> dict[str, "torch.Tensor"]:
        batch = tuple(batch)
        keys = batch[0].keys()
        batch = tuple(zip(*[b.values() for b in batch]))
        return {
            k: torch.stack(batch[i]) for i, k in enumerate(keys)
        }


class DomainDataset(Dataset):
    def __init__(self, *domains: "ImageDataset"):
        self._domains = domains

    def __len__(self):
        return max(len(d) for d in self._domains)

    def __getitem__(self, item):
        return {
            f"domain_{i}": d[item] for i, d in enumerate(self._domains)
        }


__all__ = [
    "ImageDataset",
    "DomainDataset"
]
