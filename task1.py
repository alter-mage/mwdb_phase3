import pickle
import os
import utilities
import generate_data_matrix
import svm
import numpy as np
import cv2
import csv
import min_max_scaler
import ppr
import decision_tree
import accuracy_metrics
from min_max_scaler import transform as min_max_tsf
from sklearn.model_selection import train_test_split

from svm import MulticlassSVM as SVM

def start_task1():
    with open('metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)
    with open('simp.pickle', 'rb') as handle:
        simp_data = pickle.load(handle)
    
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
    

    if model == 0:
        reduction_technique = 0
    elif model == 1:
        reduction_technique = 1
    else:
        reduction_technique = 2


    data_matrix, label_matrix = generate_data_matrix.get_matrix(metadata, model, 1)
    data_matrix = min_max_tsf(data_matrix)


    reduction_obj_right = utilities.reduction_technique_map[reduction_technique](k, data_matrix)
    left_matrix, core_matrix, right_matrix = reduction_obj_right.transform()
    X_train, Y_train = left_matrix, label_matrix


    # Getting features of test_images
    test_dir = os.path.join(os.getcwd(), 'test_images')
    if not os.path.isdir(test_dir):
        print("test_images file not found!")
        quit()
    test_array = []
    test_labels = []
    for filename in os.listdir(test_dir):
        if filename in metadata:
            test_array.append(metadata[filename][utilities.feature_models[model]])
            test_labels.append(metadata[filename]['x_label'])
        else:
            img = cv2.imread(os.path.join(test_dir, filename), cv2.IMREAD_GRAYSCALE)
            test_array.append(utilities.feature_extraction[model](img))
            x, y, z = filename.split('.')[0].split('-')[1:]
            test_labels.append(utilities.label_dict[x])
    test_array = np.array(test_array)
    test_labels = np.array(test_labels)
    right_matrix = np.array(right_matrix)
    right_matrix = np.transpose(right_matrix)
    # print(test_array.shape, test_labels.shape, right_matrix.shape)
    test_array = np.dot(test_array, right_matrix)
    print(test_array.shape)
    if c == 0:
        print('\nSVM')
        # clf = svm.fit(left_matrix, label_matrix=label_matrix)
        clf = SVM()
        clf.fit(X_train,Y_train)
        # prediction = []
        # for i in range(len(test_array)):
        #     pred = int(clf.predict(test_array[i]))
        #     prediction.append(pred)
        prediction = clf.predict(test_array)
        print(prediction)
    
    elif c == 1:
        print('\nDecision Tree')
        clf = decision_tree.DecisionTreeClassifier()
        node = clf.fit(X_train,Y_train)
        prediction = []
        for i in range(len(test_array)):
            pred = int(clf.predict(test_array[i], node))
            prediction.append(pred)
    else:
        print('\nPPR')
        print(utilities.feature_models[model])
        similarity_m = simp_data[utilities.feature_models[model]]['T']
        print(similarity_m.shape, test_array.shape)
        prediction = ppr.predict(similarity_m, test_array)
    
    latent_out_file_path = '%s_%s_%s_%s' % ('1', utilities.feature_models[model], str(k), utilities.reduction_technique_map_str[reduction_technique])
    with open(latent_out_file_path+'.pickle', 'wb') as handle:
        pickle.dump({
            'left_matrix': left_matrix,
            'core_matrix': core_matrix,
            'right_matrix': right_matrix
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # fpr, fnr = accuracy_metrics.metrics(test_labels, prediction)
    count = 0
    total = 0
    for i in range(len(prediction)):
        print(prediction[i], test_labels[i])
        if test_labels[i] == prediction[i]:
            count +=1
        total +=1 
    print(count, total)
    # print(left_matrix.shape, core_matrix.shape, right_matrix.shape)
    #clf = svm.fit(left_matrix, label_matrix=label_matrix)
    # clf = decision_tree.fit(left_matrix,label_matrix=label_matrix)

    # X_train, X_test, y_train, y_test = train_test_split(left_matrix, label_matrix,test_size = 0.10)
     
    # clf = SVM()
    # clf.fit(X_train,y_train)
    # count = 0
    # total = 0
    # prediction = clf.predict(X_test)
    
    # for i in range(len(prediction)):
    #     #print(prediction[i] , y_test[i])
    #     if prediction[i] == y_test[i]:
    #         count +=1 
    #     total+=1
    # print(count , total) 
    """
    right_matrix = np.array(right_matrix)
    right_matrix = np.transpose(right_matrix)
    print(test_array.shape, test_labels.shape, right_matrix.shape)
    test_array = np.dot(test_array, right_matrix)
    #print(test_array)
    #print(test_array.shape)
    print(test_labels)
    result_array = []
    for i in range(len(test_array)):
        result_array.append(clf.predict([test_array[i]]))
    #print(result_array, test_labels[0])
    for i in range(len(result_array)):
        print(result_array[i], test_labels[i])
    #print(clf.predict([left_matrix[0]]))
    """
    
if __name__ == '__main__':
    start_task1() 