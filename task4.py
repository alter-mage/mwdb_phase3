import pickle
import numpy as np


def start_task4():
    print('here')
    l = int(input('enter num of layers: '))
    k = int(input('enter num of hashes per layer: '))
    vector_file = input('enter vector file: ')

    with open(vector_file + '.pickle', 'rb') as handle:
        latent_semantics = pickle.load(handle)
    left_vector = latent_semantics['left_vector']
    right_vector = latent_semantics['right_vector']

    hash_family = {}
    random_vectors = [np.random.randn(k, len(left_vector[0])) for _ in range(l)]
