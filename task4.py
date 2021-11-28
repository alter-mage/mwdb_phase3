import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import math

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
    retrieve_image_indexes, buckets_searched = [], 0
    for query_hash_code, hash_bucket in zip(query_hash_codes, layered_hash_buckets):
        for hash_code in hash_bucket:
            if query_hash_code[:k - pruned_hash] == hash_code[:k - pruned_hash]:
                buckets_searched += 1
                retrieve_image_indexes.extend(hash_bucket[hash_code])
    return retrieve_image_indexes, buckets_searched


def build_indexes(l, k, left_matrix):
    hash_family = {}
    random_vectors = [np.random.randn(k, len(left_matrix[0])) for _ in range(l)]
    for index, hashes in enumerate(random_vectors):
        hash_family['l' + str(index + 1)] = hashes

    return hash_family


def populate_indexes(k, hash_family, index_images, query):
    layered_hash_bucket = build_hash_buckets(hash_family, index_images)
    
    index_size = 0
    for bucket in layered_hash_bucket:
        bucket_keys = len(bucket)
        index_size += math.ceil(math.log2(k)) * bucket_keys * len(index_images)

    query_hash_codes = []
    for layer in hash_family:
        query_hash_layer_value = hash_family[layer].dot(np.array(query))
        query_hash_value_code = ''.join(['1' if hash_value > 0 else '0' for hash_value in query_hash_layer_value])
        query_hash_codes.append(query_hash_value_code)

    return layered_hash_bucket, query_hash_codes, index_size


def build_index(l, k, left_matrix, index_images, query):
    hash_family = {}
    random_vectors = [np.random.randn(l, len(left_matrix[0])) for _ in range(k)]
    for index, hashes in enumerate(random_vectors):
        hash_family['l' + str(index + 1)] = hashes

    layered_hash_bucket = build_hash_buckets(hash_family, index_images)

    query_hash_codes = []
    for layer in hash_family:
        query_hash_layer_value = hash_family[layer].dot(np.array(query))
        query_hash_value_code = ''.join(['1' if hash_value > 0 else '0' for hash_value in query_hash_layer_value])
        query_hash_codes.append(query_hash_value_code)

    return layered_hash_bucket, query_hash_codes


def get_top_images(l, k, vector_file, t, image_folder, query_image):
    if vector_file == 'all':
        feature_model_inp = int(input('enter feature model: '))
        feature_model = utilities.feature_extraction[feature_model_inp]

        query_transformation_model_inp = int(input('enter reduction model: '))
        query_transformation_model = utilities.query_transformation[query_transformation_model_inp]

        similarity_model = utilities.similarity_map[feature_model_inp]
    else:
        vector_file_tokens = vector_file.split('_')
        feature_model = utilities.feature_extraction[utilities.feature_models.index(vector_file_tokens[1])]
        query_transformation_model = utilities.query_transformation[utilities.reduction_technique_map_str.index(
            vector_file_tokens[-1]
        )]
        similarity_model = utilities.similarity_map[utilities.feature_models.index(vector_file_tokens[1])]

    images, features = [], []
    for filename in os.listdir(image_folder):
        img = cv2.imread(os.path.join(image_folder, filename), cv2.IMREAD_GRAYSCALE)
        img_vector = feature_model(img)
        images.append(img)
        features.append(img_vector)

    query_image_vector = feature_model(query_image)

    features.append(query_image_vector)
    scaled_features = min_max_scaler.transform(features)

    if vector_file == 'all':
        left_matrix = scaled_features
        right_matrix = np.full(shape=(len(scaled_features[1]), len(scaled_features[1])), fill_value=1)
    else:
        with open(vector_file + '.pickle', 'rb') as handle:
            latent_semantics = pickle.load(handle)
        left_matrix = latent_semantics['left_matrix']
        right_matrix = latent_semantics['right_matrix']

    transformed_images = query_transformation_model(scaled_features, right_matrix)

    index_images = transformed_images[:-1]
    query = transformed_images[-1]

    if vector_file == 'all':
        vector_file = '_'.join([utilities.feature_models[utilities.feature_extraction.index(feature_model)], 'all'])
    index_file_path = os.path.join(os.getcwd(), '_'.join(['lsh', str(l), str(k), vector_file]) + '.pickle')
    if not os.path.isfile(index_file_path):
        hash_family = build_indexes(l, k, left_matrix)
        with open(index_file_path, 'wb') as handle:
            pickle.dump(hash_family, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(index_file_path, 'rb') as handle:
            hash_family = pickle.load(handle)

    layered_hash_bucket, query_hash_codes, index_size = populate_indexes(k, hash_family, index_images, query)

    retrieved_image_indexes, buckets_searched, pruned_hash, all_images_retrieved = [], 0, -1, False
    while len(set(retrieved_image_indexes)) < t:
        pruned_hash += 1
        retrieved_image_indexes, buckets_searched = retrieve_images(layered_hash_bucket, query_hash_codes, pruned_hash, k)

        if len(set(retrieved_image_indexes)) == len(index_images):
            all_images_retrieved = True
            break
    unique_image_indexes = set(retrieved_image_indexes)

    unique_images = [index_images[image_index] for image_index in unique_image_indexes]
    similarity_map = similarity_model(query, unique_images)
    similarity_image_map = [[score, index_images[index], images[index]] for score, index in zip(
        similarity_map, unique_image_indexes
    )]
    similarity_image_map = sorted(similarity_image_map, key=lambda x: x[0], reverse=True)

    actual_similarity_map = similarity_model(query, index_images)
    actual_image_index_map = sorted(
        [[score, index] for index, score in enumerate(actual_similarity_map)],
        key=lambda x: x[0],
        reverse=True
    )[:t]

    hit = 0
    for actual_image in actual_image_index_map:
        if actual_image[1] in unique_image_indexes:
            hit += 1

    metrics = {
        'miss_rate': (t-hit) / t,
        'false_positive': (len(unique_image_indexes) - hit) / t,
        'index_size_in_bytes': index_size,
        'bucket_searched': buckets_searched
    }

    return similarity_image_map, metrics, all_images_retrieved


def start_task4():
    l = int(input('enter num of layers: '))
    k = int(input('enter num of hashes per layer: '))
    vector_file = input('enter vector file: ')
    image_folder = os.path.join(os.getcwd(), input('enter image folder: '))

    query_image_name = input('enter query image name')
    query_image_path = os.path.join(os.getcwd(), query_image_name + '.png')
    query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)

    t = int(input('enter number of retrievals: '))

    similarity_image_map, metrics, _ = get_top_images(
        l, k, vector_file, t, image_folder, query_image
    )

    top_images = similarity_image_map[:t]
    fig, axes = plt.subplots(t + 1, 1)
    for i, axis in enumerate(axes):
        if i == 0:
            img = query_image
            axis.text(74, 45, 'Original image', size=9)
        else:
            img = top_images[i - 1][2]
            # axis.text(74, 45, str(top_images[i - 1][0]), size=9)
        axis.imshow(img, cmap='gray')
        axis.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plt.show()

    metrics_path = os.path.join(os.getcwd(), '_'.join(['lsh', str(l), str(k), vector_file, query_image_name])+'.json')
    with open(metrics_path, 'w') as fp:
        json.dump(metrics, fp)
