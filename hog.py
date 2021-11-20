import numpy as np
import skimage.feature

def get_hog_vector(image):
    hog_vector, hog_image = skimage.feature.hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        multichannel=False
    )
    return np.array(hog_vector, dtype=np.float32).ravel()
