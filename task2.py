import pickle
import os
import utilities
import generate_data_matrix
import svm
import numpy as np
import cv2
# import aggregation
import csv
import min_max_scaler
import decision_tree
from sklearn.model_selection import train_test_split

def start_task2():

    # Reading metadata.pickle file, image representations
    with open('metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)
        
    #input model number
    model = -1
    print()
    print("Data Models:")
    for index, value in enumerate(utilities.feature_models):
        print(index, value)
    while not (0 <= model <= 2):
        model = int(input('Enter Model Number (0-2): '))
    

    #input value of k
    k_upper_limit = len(metadata[next(iter(metadata))][utilities.feature_models[model]])
    k = -1
    while not (1 <= k <= k_upper_limit - 1):
        k = int(input('Enter value of k (latent semantics): '))

    #getting data_matrix and label matrix for the task    
    data_matrix, label_matrix = generate_data_matrix.get_matrix(metadata, model, 2)
    
   
    reduction_technique = 0
    
    
    reduction_obj_right = utilities.reduction_technique_map[reduction_technique](k, data_matrix)
    left_matrix, core_matrix, right_matrix = reduction_obj_right.transform()
    # print(left_matrix.shape, core_matrix.shape, right_matrix.shape)
    #clf = svm.fit(left_matrix, label_matrix=label_matrix)
    
    X_train, X_test, y_train, y_test = train_test_split(left_matrix, label_matrix,test_size = 0.10)
    
    
    clf = decision_tree.fit(X_train,label_matrix=y_train)
    count = 0
    total = 0
    for i in range(len(X_test)):
        prediction = int(clf.predict([X_test[i]])[0]) 
        print (prediction,y_test[i])
        if prediction == y_test[i]:
            count +=1
        total +=1 

    print(count , total) 
    """
    test_dir = os.path.join(os.getcwd(), 'test_images')
    if not os.path.isdir(test_dir):
        print("test_images file not found!")
        quit()    
    test_array = []
    test_labels = []
    for filename in os.listdir(test_dir):
        if filename in metadata:
            test_array.append(metadata[filename][utilities.feature_models[model]])
            test_labels.append(metadata[filename]['y_label'])
        else:
            img = cv2.imread(os.path.join(test_dir, filename), cv2.IMREAD_GRAYSCALE)
            test_array.append(utilities.feature_extraction[model](img))
            x, y, z = filename.split('.')[0].split('-')[1:]
            test_labels.append(int(y))
    
    
    test_array = np.array(test_array)
    test_labels = np.array(test_labels)
    right_matrix = np.array(right_matrix)
    #right_matrix = np.transpose(right_matrix)
    print(test_array.shape, test_labels.shape, right_matrix.shape)
    test_array = np.dot(test_array, right_matrix)
    #print(test_array)
    #print(test_array.shape)
    result_array = []
    print(test_labels)
    for i in range(len(test_array)):
        result_array.append(clf.predict([test_array[i]]))
    #print(result_array, test_labels[0])
    count = 0
    for i in range(len(result_array)):
        print(result_array[i], test_labels[i])
        if result_array[i][0] == test_labels[i]:
            count +=1
    
    print(count)
    """
    
if __name__ == '__main__':
    start_task2() 