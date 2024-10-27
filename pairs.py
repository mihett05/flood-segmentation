import matplotlib.pyplot as plt
import numpy as np
import rasterio


def load_and_display_images_with_overlay(num_images):
    fig, axes = plt.subplots(1, num_images, figsize=(10 * num_images, 10))

    for i in range(num_images):
        # Paths for the BGR image and mask, using {i+1} in filenames
        bgr_path = f"./data/input/{i+1}.tif"
        mask_path = f"./data/output/{i+1}.tif"

        # Load BGR image and reorder to RGB
        with rasterio.open(bgr_path) as src:
            if src.count < 3:
                print(f"Image {bgr_path} does not have 3 bands for BGR.")
                continue

            # Read BGR channels and reorder to RGB
            bgr_image = src.read([1, 2, 3])
            rgb_image = np.transpose(
                bgr_image, (1, 2, 0)
            )  # Transpose to (H, W, C)
            rgb_image = rgb_image[:, :, ::-1]  # Reorder BGR to RGB

            # Apply percentile clipping for better visualization
            lower_percentile, upper_percentile = np.percentile(
                rgb_image, (2, 98)
            )
            rgb_image = np.clip(rgb_image, lower_percentile, upper_percentile)
            rgb_image = (
                (rgb_image - lower_percentile)
                / (upper_percentile - lower_percentile)
                * 255
            ).astype(np.uint8)

        # Load mask
        with rasterio.open(mask_path) as src:
            mask = src.read(1)  # Assuming the mask is a single-channel image

        # Plot RGB image with overlayed mask
        axes[i].imshow(rgb_image)
        axes[i].imshow(
            mask, cmap="jet", alpha=0.4
        )  # Overlay mask with transparency
        axes[i].set_title(f"Image {i+1} with Mask Overlay")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# Call the function with the number of image-mask pairs you want to display
load_and_display_images_with_overlay(num_images=4)
