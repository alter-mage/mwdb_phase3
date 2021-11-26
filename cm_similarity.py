import numpy as np


def get_similarity(x1, x2):
    similarities = []
    for row in x2:
        similarities.append(1 / (np.linalg.norm(np.subtract(x1, row), ord=1)))
    return similarities
