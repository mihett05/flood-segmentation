import os
import random
import shutil
from pathlib import Path


def split_dataset(
    images_dir,
    masks_dir,
    output_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
):
    random.seed(seed)

    assert train_ratio + val_ratio + test_ratio == 1, "invalid ratios"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    subsets = ["train", "val", "test"]
    for subset in subsets:
        Path(output_dir, subset, "images").mkdir(parents=True, exist_ok=True)
        Path(output_dir, subset, "masks").mkdir(parents=True, exist_ok=True)

    images = sorted(os.listdir(images_dir))
    masks = sorted(os.listdir(masks_dir))
    assert len(images) == len(masks), "images != masks"

    dataset = list(zip(images, masks))
    random.shuffle(dataset)

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_set = dataset[:train_size]
    val_set = dataset[train_size : train_size + val_size]
    test_set = dataset[train_size + val_size :]

    def copy_files(file_set, subset_name):
        for img_file, mask_file in file_set:
            shutil.copy2(
                os.path.join(images_dir, img_file),
                os.path.join(output_dir, subset_name, "images", img_file),
            )
            shutil.copy2(
                os.path.join(masks_dir, mask_file),
                os.path.join(output_dir, subset_name, "masks", mask_file),
            )

    copy_files(train_set, "train")
    copy_files(val_set, "val")
    copy_files(test_set, "test")

    print(
        f"Dataset split complete. Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}"
    )


# Usage
images_dir = "./data/images"
masks_dir = "./data/masks"
output_dir = "./dataset"

split_dataset(images_dir, masks_dir, output_dir)
