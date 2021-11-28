import numpy as np
import skimage.feature


def get_elbp_vector(image):
    elbp_vector = []

    image_dimensions = [int(len(image) / 8), int(len(image[0]) / 8)]
    for i in range(image_dimensions[0]):
        block_vector = []
        for j in range(image_dimensions[1]):
            curr_block = image[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            rotation_invariant_lbp = skimage.feature.local_binary_pattern(
                curr_block,
                P=8,
                R=2,
                method='ror'
            )
            elbp_histogram, _ = np.histogram(
                rotation_invariant_lbp,
                bins=np.arange(10)
            )
            elbp_vector.append(elbp_histogram)
    return np.array(elbp_vector).ravel()