import utilities
import numpy as np
import pickle
import os
import cv2
from min_max_scaler import transform as min_max_tsf

def task_input(metadata):
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
        
    c = -1
    print()
    print("Classifiers:")
    for index, value in enumerate(utilities.classifiers):
        print(index, value)
    while not (0 <= c <= 2):
        c = int(input('Enter Classifier Number (0-2): '))
    
    print('Test Folder:')
    test_folder = str(input('Enter the name of the test folder: '))
    
    return k, model, c, test_folder

def get_matrix(metadata, feature_model, task_number):
    data_matrix = []
    label_matrix = []
    for key in metadata:
        data_matrix.append(metadata[key][utilities.feature_models[feature_model]])
        label_matrix.append(metadata[key][utilities.labels[task_number-1]])
    return np.array(data_matrix), np.array(label_matrix)

def get_data_for_task(task_number):

    with open('metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)

    k, model, c, test_folder = task_input(metadata)

    if model == 0:
        reduction_technique = 0
    elif model == 1:
        reduction_technique = 1
    else:
        reduction_technique = 2
    
    data_matrix, label_matrix = get_matrix(metadata, model, task_number)
    test_dir = os.path.join(os.getcwd(), test_folder)
    if not os.path.isdir(test_dir):
        print("test_images file not found!")
        quit()
    
    reduction_obj_right = utilities.reduction_technique_map[reduction_technique](k, data_matrix)
    left_matrix, core_matrix, right_matrix = reduction_obj_right.transform()

    latent_out_file_path = '%s_%s_%s_%s' % (task_number, utilities.feature_models[model], str(k), utilities.reduction_technique_map_str[reduction_technique])
    with open(latent_out_file_path+'.pickle', 'wb') as handle:
        pickle.dump({
            'left_matrix': left_matrix,
            'core_matrix': core_matrix,
            'right_matrix': right_matrix
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)

    test_array = []
    test_labels = []
    for filename in os.listdir(test_dir):
        if filename in metadata:
            test_array.append(metadata[filename][utilities.feature_models[model]])
            test_labels.append(metadata[filename][utilities.labels[task_number-1]])
        else:
            img = cv2.imread(os.path.join(test_dir, filename), cv2.IMREAD_GRAYSCALE)
            test_array.append(utilities.feature_extraction[model](img))
            x, y, z = filename.split('.')[0].split('-')[1:]
            label_1 = [x,y,z]
            if task_number == 1:
                test_labels.append(utilities.label_dict[label_1[task_number-1]])
            elif task_number == 2:
                test_labels.append(int(y))
            elif task_number == 3:
                test_labels.append(int(z))
            else:
                print("Wrong Task Number\n")
            

    

    len_data_matrix = len(data_matrix)
    combined = np.vstack([data_matrix, test_array])
    combined = min_max_tsf(combined)

    reduction_obj_right = utilities.reduction_technique_map[reduction_technique](k, combined)
    left_matrix, core_matrix, right_matrix = reduction_obj_right.transform()
    
    X_train, X_test = left_matrix[:len_data_matrix], left_matrix[len_data_matrix:]
    return X_train, X_test, label_matrix, test_labels, k, model, c, test_array

if __name__ == "__main__":
    
    
    # task = -1
    # print('\n Task Number: ')
    # while not (1 <= task <= 3):
    #     task = int(input('Enter Task Number (1-3): '))

    get_matrix()