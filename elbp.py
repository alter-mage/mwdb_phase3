import numpy as np
import skimage.feature

def get_elbp_vector(image):
    rotation_invariant_lbp = skimage.feature.local_binary_pattern(
        image,
        P=8,
        R=2,
        method='ror'
    )
    elbp_histogram, _ = np.histogram(
        rotation_invariant_lbp,
        bins=np.arange(10)
    )
    return elbp_histogram