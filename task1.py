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
from get_task_data import get_data_for_task

from svm import MulticlassSVM as SVM

def start_task1():
    
    with open('simp.pickle', 'rb') as handle:
        simp_data = pickle.load(handle)
    
    
    X_train, X_test, Y_train, Y_test, k, model, c, test_array = get_data_for_task(1)
    if c == 0:
        # print('\nSVM')
        clf = SVM()
        clf.fit(X_train,Y_train)
        prediction = clf.predict(X_test)
    
    elif c == 1:
        # print('\nDecision Tree')
        clf = decision_tree.DecisionTreeClassifier()
        node = clf.fit(X_train,Y_train)
        prediction = []
        for i in range(len(X_test)):
            pred = int(clf.predict(X_test[i], node))
            prediction.append(pred)
    else:
        # print('\nPPR')
        print(utilities.feature_models[model])
        similarity_m = simp_data[utilities.feature_models[model]]['T']
        # print(similarity_m.shape, test_array.shape)
        prediction = ppr.predict(similarity_m, test_array)
    
    task_number = 1
    cl = ['SVM', 'DT', 'PPR']
    print('\nTask: {}'.format(task_number))
    print('Feature Model: {}'.format(utilities.feature_models[model]))
    print('No of Latent Semantics k: {}'.format(k))
    print('Classifier: {}\n'.format(cl[c]))




    fpr, fnr = accuracy_metrics.metrics(Y_test, prediction, 1)
    print("\n\tFalse Positive Rate \t Miss Rate\n")
    for i in range(len(fpr)):
        label = utilities.valid_x[i]
        if i == 5 or i == 6 or i == 7 or i == 11:
            print("{} {} \t\t\t {}".format(label, fpr[i], fnr[i]))
        else:
            print("{} \t {} \t\t\t {}".format(label, fpr[i], fnr[i]))
    print('\n\n')
    # results = ['Type', 'False Positive Rate', 'Miss Rate']
    # results = np.hstack([results, fpr])
    # results = np.hstack([results, fpr])
    labels = utilities.valid_x
    # labels = np.insert(labels, 0, 'Type', axis=0)
    # fpr = np.insert(fpr, 0, 'False Positive Rate', axis=0)
    # l1 = np.array(['False Positive Rate'])
    # fpr = np.concatenate((l1, fpr), axis=0)
    # # fnr = np.insert(fnr, 0, 'Miss Rate', axis=0)
    # l1 = np.array(['Miss Rate'])
    # fnr = np.concatenate((l1, fnr), axis=0)
    fpr = np.array(fpr).T
    fnr = np.array(fnr).T
    results = np.vstack((fpr, fnr))
    results = results.T
    
    out_file_path = '%s_%s_%s_%s' % (str(task_number), utilities.feature_models[model], str(k), cl[c])
    np.savetxt(out_file_path+'.csv', results, delimiter=",", fmt="%f")


    
    
if __name__ == '__main__':
    start_task1() 