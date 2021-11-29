import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import math
import sys
from operator import itemgetter
from itertools import islice

import min_max_scaler
import utilities

def get_top_images(input_list=None):
    if not input_list:
        b = int(input('enter num of bits per dimensions (b): '))
        vector_file = input('enter vector file: ')
        image_folder = os.path.join(os.getcwd(), input('enter image folder: '))

        query_image_name = input('enter query image name: ')
        query_image_path = os.path.join(os.getcwd(), query_image_name + '.png')
        query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)

        t = int(input('enter number of retrievals: '))

        input_list = [b, vector_file, t, image_folder, query_image_name]
    else:
        b, vector_file, t, image_folder, query_image_name =\
            input_list[0], input_list[1], input_list[2], input_list[3], input_list[4]
        query_image_path = os.path.join(os.getcwd(), query_image_name + '.png')
        query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)

    if vector_file == 'all':
        feature_model_inp = int(input('enter feature model: '))
        feature_model = utilities.feature_extraction[feature_model_inp]
        query_transformation_model = utilities.query_transformation[0]
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
        right_matrix = np.identity(len(scaled_features[1]))
    else:
        with open(vector_file + '.pickle', 'rb') as handle:
            latent_semantics = pickle.load(handle)
        right_matrix = latent_semantics['right_matrix']

    transformed_images = query_transformation_model(scaled_features, right_matrix)

    index_images = transformed_images[:-1]
    query = transformed_images[-1]

    d = len(index_images[0])
    bj = []
    for j in range(1,d+1):
        rem = 0
        if j <= b%d:
            rem = 1
        bj.append(math.floor(b/d)+rem)

    p = []
    for j in range(d):
        splits = np.array_split(sorted(index_images[:,j]),2**bj[j])
        pj = []
        for split in splits:
            pj.append(split[0])
        pj.append(splits[-1][-1]+1)
        p.append(pj)
    
    va = []
    rq = []
    for i in range(len(transformed_images)):
        hash = ''
        for j in range(d):
            for k in range(len(p[j])-1):
                if p[j][k] <= transformed_images[i][j] < p[j][k + 1]:
                    if i == len(transformed_images)-1:
                        rq.append(k)
                    elif bj[j] > 0:
                        hash += str(format(k, '0'+str(bj[j])+'b'))
                    break
        if i < len(transformed_images)-1:
            va.append(hash)
    
    dst = [(sys.maxsize, -1) for i in range(t)]
    dist = sys.maxsize
    l = {}
    images_considered = []
    for i in range(len(index_images)):
        ai = va[i]
        it = iter(ai)
        ri = [''.join(islice(it, None, x)) for x in bj[:b]]
        ri = [int(rij,2) for rij in ri]
        if len(ri) < d:
            for _ in range(d-len(ri)):
                ri.append(0)
        if tuple(ri) in l:
            li = l[tuple(ri)]
        else:
            li = 0
            for j in range(d):
                if ri[j] < rq[j]:
                    li += query[j] - p[j][ri[j]+1]
                elif ri[j] > rq[j]:
                    li += p[j][ri[j]] - query[j]
            l[tuple(ri)] = li
        if li < dist:
            di = np.linalg.norm((index_images[i] - query), ord=1)
            images_considered.append(i)
            if di < dst[-1][0]:
                dst[-1] = (di, i)
                dst = sorted(dst, key=itemgetter(0))
            dist = dst[-1][0]

    actual = similarity_model(query, index_images)
    actual = [(actual[i], i) for i in range(len(actual))]
    actual = sorted(actual, key=itemgetter(0), reverse=True)[:t]
    actual_ind = [actuali[1] for actuali in actual]
    hit = 0
    for i in actual_ind:
        if i in images_considered:
            hit += 1
    
    metrics = {
        'index_size_in_bytes': math.ceil(math.log2(b))*len(index_images),
        'bucket_searched': len(l),
        'unique_images_considered': len(images_considered),
        'overall_images_considered': len(images_considered),
        'miss_rate': (t - hit) / t,
        'false_positive_rate': (len(images_considered) - hit) / t
    }

    top_images = [[dst[i][0], index_images[[dst[i][1]]], images[dst[i][1]]] for i in range(len(dst))]
    return input_list, top_images, metrics, False


def start_task5():
    input_list, top_images, metrics, _ = get_top_images()
    b, vector_file, t, image_folder, query_image_name = \
        input_list[0], input_list[1], input_list[2], input_list[3], input_list[4]
    query_image_path = os.path.join(os.getcwd(), query_image_name + '.png')
    query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)

    fig, axes = plt.subplots(t + 1, 1)
    for i, axis in enumerate(axes):
        if i == 0:
            img = query_image
            axis.text(74, 45, 'Original image', size=9)
        else:
            img = top_images[i - 1]
        axis.imshow(img, cmap='gray')
        axis.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plt.show()

    metrics_path = os.path.join(os.getcwd(), '_'.join(['va', str(b), vector_file, query_image_name])+'.json')
    with open(metrics_path, 'w') as fp:
        json.dump(metrics, fp)


if __name__ == '__main__':
    start_task5()
