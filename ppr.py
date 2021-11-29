import math
import numpy as np
import pickle
import os
import utilities
import cv2
import min_max_scaler

def predict(similarity_m, test_array):
        num_subjects = len(similarity_m) + 1
        c = 0.5
        pagerank = np.random.uniform(low=0, high=1, size=num_subjects)
        s_vector = np.zeros(num_subjects)
        s_vector[-1] = 1
        results = []
        for i in range(len(test_array)):
                q = np.array(test_array[i])
                q = q.T
                similarity_q_m = np.vstack([similarity_m, q])
                t_matrix = similarity_q_m @ similarity_q_m.T
                similarity_m_1 = min_max_scaler.transform(t_matrix)
                pagerank = np.linalg.inv(np.identity(num_subjects) - (1-c) * similarity_m_1) @ np.atleast_2d(c * s_vector).T
                pagerank = pagerank[:-1]
                # pagerank = np.argsort(pagerank)[::-1]
                cl = np.argmax(pagerank)
                results.append(cl)
        return results