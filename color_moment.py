import math
import numpy as np
import scipy


def get_cm_vector(image):
    cm_vector=[]
    image_dimensions = [int(len(image) / 8), int(len(image[0]) / 8)]
    for i in range(image_dimensions[0]):
        block_vector = []
        for j in range(image_dimensions[1]):
            curr_block = image[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8].ravel()
            mean, std, skew = np.mean(curr_block), math.sqrt(np.var(curr_block)), scipy.stats.skew(curr_block)
            block_vector.append([mean, std, skew])
        cm_vector.append(block_vector)
    return np.array(cm_vector, dtype=np.float32).ravel()

