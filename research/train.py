import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from torchgeo.models import ResNet18_Weights, resnet18
from torchgeo.trainers import SemanticSegmentationTask

from dataloader import create_dataloader

dataloader = create_dataloader("./data")
print(dataloader.dataset.files)
for batch in dataloader:
    print(dataloader)

# Load the pretrained ResNet18 backbone with 13 channels
backbone = resnet18(weights=ResNet18_Weights.SENTINEL2_ALL_MOCO)
original_conv = backbone.conv1

# Modify the input layer to accept 10 channels
new_conv = nn.Conv2d(
    in_channels=10,
    out_channels=original_conv.out_channels,
    kernel_size=original_conv.kernel_size,
    stride=original_conv.stride,
    padding=original_conv.padding,
    bias=original_conv.bias is not None,
)

# Adjust weights from 13 to 10 channels by slicing
new_conv.weight.data = original_conv.weight.data[:, :10, :, :]
if original_conv.bias is not None:
    new_conv.bias.data = original_conv.bias.data

# Initialize the SemanticSegmentationTask with "unet" model
task = SemanticSegmentationTask(
    model="unet",
    loss="ce",
    in_channels=10,
    lr=0.001,
    patience=5,
)

# Replace the encoder's first layer with the modified layer
task.model.encoder.conv1 = new_conv

# Set up the Trainer
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
trainer = Trainer(
    accelerator=accelerator,
    default_root_dir="./exp",
    fast_dev_run=False,
    log_every_n_steps=1,
    min_epochs=1,
    max_epochs=10,
    check_val_every_n_epoch=None,
)

# Start training
trainer.fit(model=task, train_dataloaders=dataloader)
trainer.predict()