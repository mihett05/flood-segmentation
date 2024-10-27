import os
from typing import List, Optional

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
from tqdm import tqdm


def get_tiles_with_overlap(
    image_width: int, image_height: int, tile_size: int, overlap: int
) -> List[Window]:
    """
    Calculate the windows for tiles with specified overlap across the image.

    Parameters:
        image_width (int): The width of the input image in pixels.
        image_height (int): The height of the input image in pixels.
        tile_size (int): The size of each tile (assumes square tiles).
        overlap (int): The number of overlapping pixels between adjacent tiles.

    Returns:
        List[Window]: A list of rasterio Window objects representing each tile.
    """
    step_size = tile_size - overlap
    tiles = []
    for y in range(0, image_height, step_size):
        for x in range(0, image_width, step_size):
            window = Window(x, y, tile_size, tile_size)
            # Adjust window if it exceeds the image bounds
            window = window.intersection(
                Window(0, 0, image_width, image_height)
            )
            tiles.append(window)
    return tiles


def save_tile(
    src_dataset: rasterio.io.DatasetReader,
    window: Window,
    output_folder: str,
    tile_index: int,
    image_id: str,
) -> None:
    """
    Extract and save a single tile from the source dataset.

    Parameters:
        src_dataset (rasterio.io.DatasetReader): The opened rasterio dataset (the input image).
        window (Window): The window (rasterio Window object) defining the tile.
        output_folder (str): The folder where the tiles will be saved.
        tile_index (int): Index of the tile to be used for naming the file.
        image_id (int): Image id to be used for naming the file.

    Returns:
        None
    """
    transform = src_dataset.window_transform(window)
    tile_data = src_dataset.read(window=window)

    profile = src_dataset.profile
    profile.update(
        {
            "driver": "GTiff",
            "height": window.height,
            "width": window.width,
            "transform": transform,
        }
    )

    output_filename = os.path.join(
        output_folder, f"tile_{image_id}_{tile_index}.tif"
    )
    with rasterio.open(output_filename, "w", **profile) as dst:
        dst.write(tile_data)


def split_image(
    image_path: str,
    output_folder: str,
    mask_path: Optional[str] = None,
    tile_size: int = 512,
    overlap: int = 128,
    image_id: str = "0",
):
    """
    Split a large GeoTIFF image and its corresponding mask (if provided) into tiles with overlap
    and save them.

    Parameters:
        image_path (str): The file path of the input TIFF image.
        mask_path (Optional[str]): The file path of the corresponding mask TIFF image. If None, only image is processed.
        output_folder (str): The folder where the tiles will be saved.
        tile_size (int, optional): The size of the tiles. Default is 512x512.
        overlap (int, optional): The number of pixels to overlap between tiles. Default is 128 pixels.
        image_id (int, optional): ID of the input image to be used for naming the file.
            Defaults to 0.

    Returns:
        None
    """
    with rasterio.open(image_path) as src_image:
        image_width = src_image.width
        image_height = src_image.height

        # Create output directories for images and masks (if available)
        images_folder = os.path.join(output_folder, "images")
        os.makedirs(images_folder, exist_ok=True)

        if mask_path:
            masks_folder = os.path.join(output_folder, "masks")
            os.makedirs(masks_folder, exist_ok=True)

        # Get list of tiles with overlap
        tiles = get_tiles_with_overlap(
            image_width, image_height, tile_size, overlap
        )

        # Save image tiles (and mask tiles if provided)
        if mask_path:
            with rasterio.open(mask_path) as src_mask:
                for idx, window in tqdm(enumerate(tiles)):
                    save_tile(src_image, window, images_folder, idx, image_id)
                    save_tile(src_mask, window, masks_folder, idx, image_id)
        else:
            for idx, window in tqdm(enumerate(tiles)):
                save_tile(src_image, window, images_folder, idx, image_id)
        return src_image.profile


def merge_tiles(
    tiles_folder: str,
    output_path: str,
    tile_size: int = 512,
    overlap: int = 128,
    src_profile=None,
) -> None:
    """
    Merge image tiles into a single GeoTIFF file, considering overlapping regions.

    Parameters:
        tiles_folder (str): The folder containing the image tiles to be merged.
        output_path (str): The file path of the output merged GeoTIFF.
        tile_size (int, optional): The size of the tiles. Default is 512x512.
        overlap (int, optional): The number of pixels that overlap between tiles. Default is 128 pixels.

    Returns:
        None
    """
    # Gather all tile files in the folder
    tile_files = [
        os.path.join(tiles_folder, file_name)
        for file_name in os.listdir(tiles_folder)
        if file_name.endswith(".tif")
    ]

    # Open all tiles with rasterio and add them to the list of sources
    sources = [rasterio.open(tile_file) for tile_file in tile_files]

    # Determine the bounds of the final merged image
    min_x, min_y, max_x, max_y = None, None, None, None
    for src in sources:
        bounds = src.bounds
        min_x = bounds.left if min_x is None else min(min_x, bounds.left)
        min_y = bounds.bottom if min_y is None else min(min_y, bounds.bottom)
        max_x = bounds.right if max_x is None else max(max_x, bounds.right)
        max_y = bounds.top if max_y is None else max(max_y, bounds.top)

    # Calculate the shape of the merged image, considering tile size and overlap
    pixel_size_x = sources[0].transform[0]
    pixel_size_y = -sources[0].transform[4]
    merged_width = int((max_x - min_x) / pixel_size_x)
    merged_height = int((max_y - min_y) / pixel_size_y)
    merged_transform = from_origin(min_x, max_y, pixel_size_x, pixel_size_y)

    # Create arrays to hold the merged data and the count of contributions
    merged_data = np.zeros(
        (sources[0].count, merged_height, merged_width), dtype=np.float32
    )
    count_data = np.zeros((merged_height, merged_width), dtype=np.float32)

    # Iterate over each source and add its data to the merged arrays, considering tile size and overlap
    for src in sources:
        tile_data = src.read()
        col_off = int((src.bounds.left - min_x) / pixel_size_x)
        row_off = int((max_y - src.bounds.top) / pixel_size_y)
        height, width = tile_data.shape[1], tile_data.shape[2]

        # Adjust for overlap and tile size
        row_start = max(0, row_off)
        row_end = min(merged_height, row_off + tile_size)
        col_start = max(0, col_off)
        col_end = min(merged_width, col_off + tile_size)

        tile_row_end = min(height, row_end - row_start)
        tile_col_end = min(width, col_end - col_start)

        merged_data[
            :,
            row_start : row_start + tile_row_end,
            col_start : col_start + tile_col_end,
        ] += tile_data[:, :tile_row_end, :tile_col_end]
        count_data[
            row_start : row_start + tile_row_end,
            col_start : col_start + tile_col_end,
        ] += 1

    # Avoid black lines by averaging overlapping pixels
    count_data[count_data == 0] = 1
    merged_data = (merged_data / count_data).astype(np.uint8)
    merged_data = np.where(merged_data >= 127.5, 255, 0).astype(np.uint16)

    # Use the profile from the first tile to set up the output file
    profile = sources[0].profile
    profile.update(
        {
            "driver": "GTiff",
            "nodata": src_profile["nodata"],
            "height": merged_height,
            "width": merged_width,
            "transform": src_profile["transform"],
            "dtype": "uint16",
            "count": merged_data.shape[0],
            "bounds": (min_x, min_y, max_x, max_y),
            "crs": src_profile["crs"],
            "blockysize": src_profile["blockysize"],
            "tiled": src_profile["tiled"],
            "interleave": src_profile["interleave"],
        }
    )

    # Write the merged data to the output file
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(merged_data)

    # Close all source files
    for src in sources:
        src.close()


if __name__ == "__main__":
    for num in ["1", "2", "4", "5", "6_1", "6_2", "9_1", "9_2"]:
        image_path = f"./train/images/{num}.tif"
        mask_path = f"./train/masks/{num}.tif"

        output_folder = "data/"

        split_image(
            image_path=image_path,
            mask_path=mask_path,
            output_folder=output_folder,
            tile_size=256,
            overlap=32,
            image_id=num,
        )
