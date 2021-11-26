import cv2
import scipy
import scipy.stats
import numpy as np


def get_similarity(x1, x2):
    x1_row = x1
    if len(x1.shape) == 2:
        x1_row = np.reshape(x1, x1.shape[1])
    similarities = []
    for row in x2:
        similarities.append(1 / scipy.stats.wasserstein_distance(x1_row, row))
    return similarities