import pickle
import utilities
import generate_data_matrix
import svm
# import aggregation
import csv
import min_max_scaler

def start_task1():
    # Reading metadata.pickle file, image representations
    with open('metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)
    
    model = -1
    print()
    print("Data Models:")
    for index, value in enumerate(utilities.feature_models):
        print(index, value)
    while not (0 <= model <= 2):
        model = int(input('Enter Model Number (0-2): '))
    
    k_upper_limit = len(metadata[next(iter(metadata))][utilities.feature_models[model]])
    print()
    k = -1
    while not (1 <= k <= k_upper_limit - 1):
        k = int(input('Enter value of k (latent semantics): '))
    data_matrix, label_matrix = generate_data_matrix.get_matrix(metadata, model, 1)
    # print(data_matrix.shape, label_matrix.shape)
    if model == 0:
        reduction_technique = 0
    elif model == 1:
        reduction_technique = 1
    else:
        reduction_technique = 2
    
    reduction_obj_right = utilities.reduction_technique_map[reduction_technique](k, data_matrix)
    left_matrix, core_matrix, right_matrix = reduction_obj_right.transform()
    # print(left_matrix.shape, core_matrix.shape, right_matrix.shape)
    clf = svm.fit(left_matrix, label_matrix=label_matrix)
    print(clf.predict([left_matrix[0]]))
    
if __name__ == '__main__':
    start_task1() 