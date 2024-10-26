import os
from random import randint

import planetary_computer
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import (
    NAIP,
    ChesapeakeDE,
    IntersectionDataset,
    RasterDataset,
    stack_samples,
)
from torchgeo.datasets.utils import download_url
from torchgeo.samplers import RandomGeoSampler


class Sentinel2(RasterDataset):
    filename_glob = "*.tif"
    is_image = True
    separate_files = False
    rgb_bands = ("B04", "B03", "B02")
    all_bands = (
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B08A",
        "B11",
        "B12",
    )


class MasksDataset(RasterDataset):
    filename_glob = "*.tif"
    is_image = False
    separate_files = False
    all_bands = ("B01",)


torch.manual_seed(randint(1, 1000))


def custom_collate(sample):
    sample = stack_samples(sample)
    return {
        "image": sample["image"] if "image" in sample else None,
        "mask": sample["mask"] if "mask" in sample else None,
    }


def create_dataloader(
    images: str = ".\\train\\images", masks: str = ".\\train\\masks"
) -> DataLoader:
    dataset = Sentinel2(images)
    masks = MasksDataset(masks)
    sampler = RandomGeoSampler(dataset, size=128, length=3)

    dataloader = DataLoader(
        IntersectionDataset(images, masks),
        sampler=sampler,
        collate_fn=custom_collate,
    )
    return dataloader


def download():
    naip_root = "./naip"

    naip_url = "https://naipeuwest.blob.core.windows.net/naip/v002/de/2018/de_060cm_2018/38075/"
    tiles = [
        "m_3807511_ne_18_060_20181104.tif",
        "m_3807511_se_18_060_20181104.tif",
        "m_3807512_nw_18_060_20180815.tif",
        "m_3807512_sw_18_060_20180815.tif",
    ]
    for tile in tiles:
        url = planetary_computer.sign(naip_url + tile)
        download_url(url, naip_root, filename=tile)

    naip = NAIP(naip_root)

    chesapeake_root = os.path.join("./chesapeake")

    os.makedirs(chesapeake_root, exist_ok=True)
    chesapeake = ChesapeakeDE(
        chesapeake_root, crs=naip.crs, res=naip.res, download=True
    )

    dataset = naip & chesapeake


if __name__ == "__main__":
    dataset = download()
    sampler = RandomGeoSampler(dataset, size=1000, length=10)

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=stack_samples,
    )

    for sample in dataloader:
        print(sample["image"], sample["mask"])
