import numpy as np
import utilities
import min_max_scaler


def group_by_type(metadata, y, feature_model):
    type_image_map = {}
    for key in sorted(metadata):
        key_tokens = key.split('.')[0].split('-')
        if int(key_tokens[2]) == y:
            if key_tokens[1] not in type_image_map:
                type_image_map[key_tokens[1]] = []
            type_image_map[key_tokens[1]].append(
                metadata[key][utilities.feature_models[feature_model]]
            )

    data_matrix, data_matrix_type_index_map, types, data_matrix_index = [], {}, [], 0
    for type_ in sorted(type_image_map):
        data_matrix += type_image_map[type_]
        data_matrix_type_index_map[type_] = [data_matrix_index,
                                             data_matrix_index + len(type_image_map[type_])]
        data_matrix_index += len(type_image_map[type_])
        types.append(str(type_))
    return min_max_scaler.transform(data_matrix), types, data_matrix_type_index_map


def group_by_type_all(metadata, feature_model):
    type_image_map = {}
    for key in metadata:
        key_tokens = key.split('.')[0].split('-')
        if key_tokens[1] not in type_image_map:
            type_image_map[key_tokens[1]] = []
        type_image_map[key_tokens[1]].append(
            metadata[key][utilities.feature_models[feature_model]]
        )
    data_matrix = []
    for type_ in sorted(type_image_map):
        data_matrix.append(np.mean(type_image_map[type_], axis=0))
    data_matrix = np.array(min_max_scaler.transform(data_matrix), dtype=np.float32)
    type_matrix = min_max_scaler.transform(data_matrix)
    return sorted(type_image_map), type_matrix


def group_by_subject(metadata, x, feature_model):
    subject_image_map = {}
    for key in sorted(metadata):
        key_tokens = key.split('.')[0].split('-')
        if key_tokens[1] == x:
            if int(key_tokens[2]) not in subject_image_map:
                subject_image_map[int(key_tokens[2])] = []
            subject_image_map[int(key_tokens[2])].append(
                metadata[key][utilities.feature_models[feature_model]]
            )

    data_matrix, data_matrix_subject_index_map, subjects, data_matrix_index = [], {}, [], 0
    for subject in sorted(subject_image_map):
        data_matrix += subject_image_map[subject]
        data_matrix_subject_index_map[subject] = [data_matrix_index, data_matrix_index+len(subject_image_map[subject])]
        data_matrix_index += len(subject_image_map[subject])
        subjects.append(str(subject))
    return min_max_scaler.transform(data_matrix), subjects, data_matrix_subject_index_map


def group_by_subject_all(metadata, feature_model):
    subject_image_map = {}
    for key in metadata:
        key_tokens = key.split('.')[0].split('-')
        if int(key_tokens[2]) not in subject_image_map:
            subject_image_map[int(key_tokens[2])] = []
        subject_image_map[int(key_tokens[2])].append(
            metadata[key][utilities.feature_models[feature_model]])
    data_matrix = []
    for subject in sorted(subject_image_map):
        data_matrix.append(np.mean(subject_image_map[subject], axis=0))
    data_matrix = np.array(min_max_scaler.transform(data_matrix))
    subject_matrix = min_max_scaler.transform(data_matrix)
    return sorted(subject_image_map), subject_matrix


def group_by_image_all(metadata, feature_model):
    image_image_map = {}
    for key in metadata:
        key_tokens = key.split('.')[0].split('-')
        if key_tokens[3] not in image_image_map:
            image_image_map[key_tokens[3]] = []
        image_image_map[key_tokens[3]].append(
            metadata[key][utilities.feature_models[feature_model]]
        )
    data_matrix = []
    for image_ in sorted(image_image_map):
        data_matrix.append(np.mean(image_image_map[image_], axis=0))
    data_matrix = np.array(min_max_scaler.transform(data_matrix), dtype=np.float32)
    image_matrix = min_max_scaler.transform(data_matrix)
    return sorted(image_image_map), image_matrix


def all_data(metadata, query_features, feature_model):
    data_matrix = []
    for key in sorted(metadata):
        data_matrix.append(metadata[key][utilities.feature_models[feature_model]])
    data_matrix.append(query_features)
    data_matrix = np.array(min_max_scaler.transform(data_matrix))
    return data_matrix


class aggregation:

    def __init__(self):
        pass


def aggregate_by_mean(left_matrix, data_matrix_index_map):
    left_matrix_shape = left_matrix.shape
    left_matrix_aggregated = np.zeros((len(data_matrix_index_map), left_matrix_shape[1]), dtype=np.float32)
    for i, key in enumerate(data_matrix_index_map):
        curr = left_matrix[data_matrix_index_map[key][0]: data_matrix_index_map[key][1]]
        for column in range(left_matrix_shape[1]):
            left_matrix_aggregated[i][column] = np.mean(curr[:, column])
    return left_matrix_aggregated
