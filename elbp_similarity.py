import scipy.spatial


def get_similarity(x1, x2):
    similarities = []
    for row in x2:
        similarities.append(1 - scipy.spatial.distance.cosine(x1, row))
    return similarities