import os
from pathlib import Path
from typing import Union, Generator

from utils.datasets import ImageDataset


class CelebADataset(ImageDataset):
    @classmethod
    def download(cls, DIR: Union[str, "Path"]):
        cls.kaggle_download(DIR, "jessicali9530/celeba-dataset")

    def get_data(self) -> Generator[dict[str, ...], None, None]:
        files = os.listdir(folder := self.DIR / "img_align_celeba" / "img_align_celeba")
        for file in files: yield {
            "file": folder / file,
        }


class ITSDataset(ImageDataset):
    SETS = "hazy", "clear", "trans"

    @classmethod
    def download(cls, DIR: Union[str, "Path"]):
        cls.kaggle_download(DIR, "balraj98/indoor-training-set-its-residestandard")

    def get_data(self) -> Generator[dict[str, ...], None, None]:
        files = os.listdir(folder := self.DIR / self.SET)
        for file in files: yield {
            "file": folder / file,
        }


class DenseHazeCVPR2019Dataset(ImageDataset):
    SETS = "hazy", "GT"

    @classmethod
    def download(cls, DIR: Union[str, "Path"]):
        cls.kaggle_download(DIR, "rajat95gupta/hazing-images-dataset-cvpr-2019")

    def get_data(self) -> str:
        files = os.listdir(folder := self.DIR / self.SET)
        for file in files: yield {
            "file": folder / file,
        }


__all__ = [
    "CelebADataset",
    "ITSDataset",
    "DenseHazeCVPR2019Dataset",
]
