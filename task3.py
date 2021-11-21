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

def start_task3():

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
    data_matrix, label_matrix = generate_data_matrix.get_matrix(metadata, model, 3)
    
   
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



    
if __name__ == '__main__':
    start_task3() 