import argparse
import os

import numpy as np
import pytorch_lightning as pl
import rasterio
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import Dice
from torchvision.transforms.functional import resize

from split_images import merge_tiles, split_image

torch.set_float32_matmul_precision("high")


class SemanticSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, target_size=256):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_size = target_size
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])

        try:
            # Load image and mask
            with rasterio.open(image_path) as img:
                image = img.read().astype(
                    "float32"
                )  # shape: (channels, height, width)
            with rasterio.open(mask_path) as msk:
                mask = msk.read(1).astype("int64")  # shape: (height, width)
        except Exception as e:
            print(f"Error loading {image_path} or {mask_path}: {e}")
            return None  # Skip this item if an error occurs

        # Convert to torch tensors
        image = torch.tensor(image)
        mask = torch.tensor(mask, dtype=torch.int64)

        # Resize if either dimension is too small
        _, height, width = image.shape
        if height < self.target_size // 2 or width < self.target_size // 2:
            image = resize(image, (self.target_size, self.target_size))
            mask = resize(
                mask.unsqueeze(0), (self.target_size, self.target_size)
            ).squeeze(0)

        # Calculate padding to reach target_size x target_size
        _, height, width = image.shape
        pad_height = max(0, self.target_size - height)
        pad_width = max(0, self.target_size - width)

        # Symmetric padding
        padding = (
            pad_width // 2,  # Left
            pad_width - pad_width // 2,  # Right
            pad_height // 2,  # Top
            pad_height - pad_height // 2,  # Bottom
        )

        # Apply padding to image and mask
        image = F.pad(image, padding, mode="reflect")
        mask = F.pad(
            mask.unsqueeze(0), padding, mode="constant", value=0
        ).squeeze(0)

        # Ensure final dimensions are exactly target_size x target_size
        image = image[:, : self.target_size, : self.target_size]
        mask = mask[: self.target_size, : self.target_size]

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask, self.image_files[idx]  # Return file name for saving


class TilesDataset(Dataset):
    def __init__(self, images_dir, transform=None, target_size=256):
        self.images_dir = images_dir
        self.transform = transform
        self.target_size = target_size
        self.image_files = sorted(os.listdir(images_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])

        try:
            # Load image
            with rasterio.open(image_path) as img:
                image = img.read().astype(
                    "float32"
                )  # shape: (channels, height, width)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None  # Skip this item if an error occurs

        # Convert to torch tensor
        image = torch.tensor(image)

        # Resize if either dimension is too small
        _, height, width = image.shape
        if height < self.target_size // 2 or width < self.target_size // 2:
            image = resize(image, (self.target_size, self.target_size))

        # Calculate padding to reach target_size x target_size
        _, height, width = image.shape
        pad_height = max(0, self.target_size - height)
        pad_width = max(0, self.target_size - width)

        # Symmetric padding
        padding = (
            pad_width // 2,  # Left
            pad_width - pad_width // 2,  # Right
            pad_height // 2,  # Top
            pad_height - pad_height // 2,  # Bottom
        )

        # Apply padding to image
        image = F.pad(image, padding, mode="reflect")

        # Ensure final dimensions are exactly target_size x target_size
        image = image[:, : self.target_size, : self.target_size]

        if self.transform:
            image = self.transform(image)

        return image, self.image_files[idx]  # Return file name for saving


# Define model and the training module (as before)
class SegmentationModel(pl.LightningModule):
    def __init__(self, model, loss_fn, lr=0.0001):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.train_dice = Dice()
        self.val_dice = Dice()
        self.test_dice = Dice()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, mask, _ = batch
        predicted = self(image)
        predicted = predicted.squeeze(1)
        loss = self.loss_fn(predicted, mask)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_dice", self.train_dice(predicted, mask), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, mask, _ = batch
        predicted = self(image)
        predicted = predicted.squeeze(1)
        loss = self.loss_fn(predicted, mask)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_dice", self.val_dice(predicted, mask), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        image, mask, _ = batch
        predicted = self(image)
        predicted = predicted.squeeze(1)
        loss = self.loss_fn(predicted, mask)
        dice_score = self.test_dice(predicted, mask)

        # Log test loss and Dice score
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_dice", dice_score, prog_bar=True)
        return {"test_loss": loss, "test_dice": dice_score}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def train():
    train_loader, val_loader = create_dataloaders()
    checkpoint_callback, segmentation_module = load_segmentation_module()

    es = EarlyStopping("val_loss", patience=5, verbose=True)

    trainer = Trainer(
        max_epochs=30,
        callbacks=[checkpoint_callback, es],
        accelerator="gpu",
    )
    trainer.fit(segmentation_module, train_loader, val_loader)
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")


def test():
    test_loader = create_test_dataloader()
    checkpoint_callback, segmentation_module = load_segmentation_module()

    trainer = Trainer(callbacks=[checkpoint_callback])
    trainer.test(model=segmentation_module, dataloaders=test_loader)


def predict(path: str):
    # TODO: fix merge line
    # TODO: add geo data in tif
    if os.path.extsep("./data/preds"):
        os.removedirs("./data/preds")
    os.makedirs("./data/preds", exist_ok=True)
    split_image(
        image_path=path,
        output_folder="./data/preds",
        tile_size=256,
        overlap=32,
        image_id=os.path.basename(path),
    )
    predict_loader = create_predict_dataloader()
    _, segmentation_module = load_segmentation_module()

    os.makedirs("./data/preds/masks", exist_ok=True)

    segmentation_module.model.eval()
    with torch.no_grad():
        for images, file_names in predict_loader:
            images = images.to("cuda")
            for i in range(len(images)):
                with rasterio.open(
                    f"./data/preds/images/{file_names[i]}"
                ) as src:
                    profile = src.profile
                    src_shape = src.read(1).shape
                image = images[i].unsqueeze(0)
                predicted = segmentation_module(image)
                predicted = resize(predicted, (src_shape[0], src_shape[1]))
                predicted = torch.sigmoid(predicted).squeeze().cpu().numpy()

                predicted_binary = (predicted > 0.5).astype(np.uint8) * 255
                with rasterio.open(
                    "./data/preds/masks/" + file_names[i],
                    "w",
                    driver="GTiff",
                    width=src_shape[1],
                    height=src_shape[0],
                    count=1,
                    dtype=predicted_binary.dtype,
                    crs=profile["crs"],
                    transform=profile["transform"],
                ) as dst:
                    dst.write(predicted_binary, 1)

    merge_tiles(
        "./data/preds/masks",
        output_path="./data/preds/image.tif",
        tile_size=256,
        overlap=32,
    )


def create_dataloaders() -> tuple[DataLoader, DataLoader]:
    train_images_dir = "./dataset/train/images"
    train_masks_dir = "./dataset/train/masks"
    val_images_dir = "./dataset/val/images"
    val_masks_dir = "./dataset/val/masks"

    train_dataset = SemanticSegmentationDataset(
        train_images_dir, train_masks_dir, transform=None
    )
    val_dataset = SemanticSegmentationDataset(
        val_images_dir, val_masks_dir, transform=None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=15,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
    )

    return train_loader, val_loader


def create_test_dataloader() -> DataLoader:
    test_images_dir = "./dataset/test/images"
    test_masks_dir = "./dataset/test/masks"
    test_dataset = SemanticSegmentationDataset(
        test_images_dir, test_masks_dir, transform=None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
    )
    return test_loader


def create_predict_dataloader() -> DataLoader:
    tiles_images_dir = "./data/preds/images"
    tiles_dataset = TilesDataset(tiles_images_dir)
    tiles_loader = DataLoader(
        tiles_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=5,
        persistent_workers=True,
    )

    return tiles_loader


def load_segmentation_module() -> tuple[ModelCheckpoint, SegmentationModel]:
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=10,
        classes=1,
    )
    loss_fn = smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        dirpath="./checkpoints",
        filename="best-checkpoint",
    )

    files = sorted(
        [
            int("".join(filter(str.isdigit, file)) or "0")
            for file in next(os.walk(checkpoint_callback.dirpath))[2]
        ]
    )

    best_model_path = checkpoint_callback.best_model_path

    if files:
        best_model_path = os.path.join(
            "checkpoints",
            f"best-checkpoint-v{files[-1]}.ckpt"
            if files[-1]
            else "best-checkpoint.ckpt",
        )

    if os.path.exists(best_model_path):
        print("Best loaded:", best_model_path)
        segmentation_module = SegmentationModel.load_from_checkpoint(
            best_model_path, model=model, loss_fn=loss_fn
        )
    else:
        segmentation_module = SegmentationModel(model=model, loss_fn=loss_fn)

    return checkpoint_callback, segmentation_module


def main():
    parser = argparse.ArgumentParser(prog="Flooded area segmentation")
    parser.add_argument(
        "action",
        type=str,
        help="Segmentation action: train | test | predict",
    )
    parser.add_argument(
        "--path", type=str, help="Path for predict", required=False
    )
    args = parser.parse_args()
    if args.action == "test":
        test()
    elif args.action == "train":
        train()
    elif args.action == "predict":
        if not args.path:
            print("[ERROR] path for predict is required")
            return
        if not os.path.exists(args.path):
            print("[ERROR] path for predict not found")
            return
        predict(args.path)


if __name__ == "__main__":
    main()
