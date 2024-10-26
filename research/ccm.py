import numpy as np
import rasterio
from skimage.feature import graycomatrix, graycoprops


def normalize(band):
    band_min, band_max = (band.min(), band.max())
    return np.asarray(((band - band_min) / (band_max - band_min)) * 255).astype(
        int
    )


def brighten(band):
    alpha = 0.13
    beta = 0
    return np.clip(alpha * band + beta, 0, 255)


# Function to compute Color Co-occurrence Matrix (CCM)
def compute_ccm(image, distance, angle):
    # Extract all available spectral channels
    channels = [image[i] for i in range(image.shape[0])]

    # Define pairs of channels for which CCM is calculated
    channel_pairs = [
        (channels[i], channels[j])
        for i in range(len(channels))
        for j in range(i, len(channels))
        if i != j
    ]
    ccm_results = {}

    for index, (ch1, ch2) in enumerate(channel_pairs):
        ch1 = normalize(ch1)
        ch2 = normalize(ch2)
        # Stack the two channels to create a 2D array with shape (rows, cols, 2)

        # Compute the gray-level co-occurrence matrix
        ccm_1 = graycomatrix(
            ch1,
            [distance],
            [angle],
            levels=256,
            symmetric=True,
            normed=True,
        )
        ccm_2 = graycomatrix(
            ch1,
            [distance],
            [angle],
            levels=256,
            symmetric=True,
            normed=True,
        )

        # Store result for current channel pair
        ccm_results[index] = ccm_1 + ccm_2

    return ccm_results


# Function to extract Haralick features from the CCM
def extract_features(ccm):
    features = {}
    for key, matrix in ccm.items():
        # Extract Haralick features
        contrast = graycoprops(matrix, "contrast")[0, 0]
        energy = graycoprops(matrix, "energy")[0, 0]
        correlation = graycoprops(matrix, "correlation")[0, 0]
        homogeneity = graycoprops(matrix, "homogeneity")[0, 0]

        # Store features for each pair of components
        features[key] = {
            "contrast": contrast,
            "energy": energy,
            "correlation": correlation,
            "homogeneity": homogeneity,
        }

    return features


# Example usage
if __name__ == "__main__":
    import rasterio

    # Load input image captured by UAV (TIFF format)
    with rasterio.open("tile_2_0.tif") as src:
        image = (
            src.read()
        )  # Read the image as an array (shape: [bands, rows, cols])

    # Compute CCM for distance 5 and angle 0 degrees
    ccm = compute_ccm(image, distance=5, angle=0)

    # Extract features from CCM
    features = extract_features(ccm)

    # Print the extracted features
    for pair, feats in features.items():
        print(f"Features for channel pair {pair}: {feats}")
