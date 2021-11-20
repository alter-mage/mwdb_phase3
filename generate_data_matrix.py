import numpy as np
import utilities

def get_matrix(metadata, feature_model, task_number):
    data_matrix = []
    label_matrix = []
    for key in metadata:
        data_matrix.append(metadata[key][utilities.feature_models[feature_model]])
        label_matrix.append(metadata[key][utilities.labels[task_number-1]])
    return np.array(data_matrix), np.array(label_matrix)