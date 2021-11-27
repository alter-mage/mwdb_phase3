import ppr
import math
import numpy as np
import pickle
import os
import utilities
import cv2
import min_max_scaler
with open('simp.pickle', 'rb') as handle:
        simp_data = pickle.load(handle)

similarity_m = simp_data['hog']['T']
num_subjects = len(similarity_m)+1
# image[utilities.feature_models[i]] = utilities.feature_extraction[i](img)
q = 1
pagerank = np.random.uniform(low=0, high=1, size=num_subjects)
pagerank_error = np.zeros(num_subjects)
c = 0.5
# s_vector = np.full(num_subjects, fill_value=(1 / num_subjects), dtype=np.float32)
# s_vector = np.zeros(num_subjects+1)
s_vector = np.zeros(num_subjects)
s_vector[-1] = 1

images = {}
images_dir = os.path.join(os.getcwd(), 'test_images')
for filename in os.listdir(images_dir):
        image = {}
        img = cv2.imread(os.path.join(images_dir, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            image = {'image': img}
            # image['image'] = img
            # image['filename'] = filename
            for i in range(3):
                image[utilities.feature_models[i]] = utilities.feature_extraction[i](img)
            images[filename] = image

images['image-cc-13-4.png'].keys()
q = images['image-cc-13-4.png']['hog']
# q.shape
# similarity_m = np.append(similarity_m, q, axis=0)
similarity_m = np.vstack([similarity_m, q])
# similarity_m.shape
T_matrix = np.dot(similarity_m, similarity_m.T)

convergence = False
similarity_m = min_max_scaler.transform(T_matrix)

print(s_vector)
while not convergence:
    pagerank_new = (1 - c) * np.dot(similarity_m, pagerank) + c * s_vector
    
    pagerank_error_new = pagerank_new - pagerank

    convergence = True
    for i, row in enumerate(pagerank_error):
        if pagerank_error_new[i] - pagerank_error[i] > 0.01:
            # RuntimeWarning: invalid value encountered in double_scalars
            convergence = False
    pagerank = pagerank_new
    pagerank_error = pagerank_error_new
print(pagerank)
