from sys import argv

import matplotlib.pyplot as plt
import rasterio

# Path to the TIFF file
tif_file_path = argv[1] if len(argv) > 1 else "input_image.tif"

n = 10


# Open the TIFF file with rasterio
with rasterio.open(tif_file_path) as dataset:
    # Check the number of channels
    n = dataset.count

    # Read all 10 channels
    channels = dataset.read()

# Plot each channel individually
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle(f"{n}-Channel TIFF Image", fontsize=16)
print(channels.shape)
for i in range(n):
    row, col = divmod(i, 5)
    ax = axes[row, col]
    ax.imshow(channels[i], cmap="gray")
    ax.set_title(f"Channel {i + 1}")
    ax.axis("off")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
