import os
import pickle
import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import min_max_scaler
import utilities


def build_hash_buckets(hash_family, images):
    layered_hash_bucket = []
    for layer in hash_family:
        layer_bucket = {}
        for index, image in enumerate(images):
            hash_layer_value = hash_family[layer].dot(np.array(image))
            hash_value_code = ''.join(['1' if hash_value > 0 else '0' for hash_value in hash_layer_value])
            if hash_value_code not in layer_bucket:
                layer_bucket[hash_value_code] = []
            layer_bucket[hash_value_code].append(index)
        layered_hash_bucket.append(layer_bucket)
    return layered_hash_bucket


def retrieve_images(layered_hash_buckets, query_hash_codes, pruned_hash, k):
    retrieve_image_indexes = []
    for query_hash_code, hash_bucket in zip(query_hash_codes, layered_hash_buckets):
        for hash_code in hash_bucket:
            if query_hash_code[:k - pruned_hash] == hash_code[:k - pruned_hash]:
                retrieve_image_indexes.extend(hash_bucket[hash_code])
    return retrieve_image_indexes


def start_task4():
    print('here')
    l = int(input('enter num of layers: '))
    k = int(input('enter num of hashes per layer: '))
    vector_file = input('enter vector file: ')
    vector_file_tokens = vector_file.split('_')
    feature_model = utilities.feature_extraction[utilities.feature_models.index(vector_file_tokens[1])]
    query_transformation_model = utilities.query_transformation[utilities.reduction_technique_map_str.index(
        vector_file_tokens[-1]
    )]
    similarity_model = utilities.similarity_map[utilities.feature_models.index(vector_file_tokens[1])]

    with open(vector_file + '.pickle', 'rb') as handle:
        latent_semantics = pickle.load(handle)
    left_matrix = latent_semantics['left_matrix']
    right_matrix = latent_semantics['right_matrix']

    images = []
    image_folder = os.path.join(os.getcwd(), input('enter image folder: '))
    for filename in os.listdir(image_folder):
        img = cv2.imread(os.path.join(image_folder, filename), cv2.IMREAD_GRAYSCALE)
        img_vector = feature_model(img)
        images.append(img_vector)

    query_image_path = os.path.join(os.getcwd(), input('enter query image name')+'.png')
    query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
    query_image_vector = feature_model(query_image)

    images.append(query_image_vector)
    # images = min_max_scaler.transform(images)
    # transformed_images = query_transformation_model(images, right_matrix)

    Scaler = MinMaxScaler()
    min_max_images = Scaler.fit_transform(images)
    transformed_images = query_transformation_model(min_max_images, right_matrix)

    index_images = transformed_images[:-1]
    query = transformed_images[-1]

    hash_family = {}
    random_vectors = [np.random.randn(k, len(left_matrix[0])) for _ in range(l)]
    for index, hashes in enumerate(random_vectors):
        hash_family['l'+str(index+1)] = hashes

    layered_hash_bucket = build_hash_buckets(hash_family, index_images)

    query_hash_codes = []
    for layer in hash_family:
        query_hash_layer_value = hash_family[layer].dot(np.array(query))
        query_hash_value_code = ''.join(['1' if hash_value > 0 else '0' for hash_value in query_hash_layer_value])
        query_hash_codes.append(query_hash_value_code)

    t = int(input('enter number of retrievals: '))
    retrieved_image_indexes, pruned_hash = [], -1
    while len(set(retrieved_image_indexes)) < t:
        pruned_hash += 1
        retrieved_image_indexes = retrieve_images(layered_hash_bucket, query_hash_codes, pruned_hash, k)
    unique_image_indexes = set(retrieved_image_indexes)

    unique_images = [index_images[image_index] for image_index in unique_image_indexes]
    image_similarity_map = similarity_model(query, unique_images)
    image_similarity_map = sorted(image_similarity_map, key=lambda x: x[0], reverse=True)[:t]

    fig, axes = plt.subplots(t + 1, 1)
    topk = []
    for i, axis in enumerate(axes):
        if i == 0:
            img = query_image
            axis.text(74, 25, query, size=9)
            axis.text(74, 45, 'Original image', size=9)
        else:
            img = image_similarity_map[i][1]
            axis.text(74, 25, images[i - 1], size=9)
            axis.text(74, 45, str(image_similarity_map[i][0]), size=9)
            topk.append(img)
        axis.imshow(img, cmap='gray')
        axis.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    fig.suptitle(str(t) + ' most similar images - ' +
                 utilities.similarity_measures[utilities.feature_models.index(vector_file_tokens[1])], size=10)
    plt.show()
    print('here')