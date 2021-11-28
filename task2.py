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

def start_task2():
    
    with open('simp.pickle', 'rb') as handle:
        simp_data = pickle.load(handle)
    
    
    X_train, X_test, Y_train, Y_test, k, model, c, test_array = get_data_for_task(2)
    if c == 0:
        print('\nSVM')
        clf = SVM()
        clf.fit(X_train,Y_train)
        prediction = clf.predict(X_test)
        print(prediction)
    
    elif c == 1:
        print('\nDecision Tree')
        clf = decision_tree.DecisionTreeClassifier()
        node = clf.fit(X_train,Y_train)
        prediction = []
        for i in range(len(X_test)):
            pred = int(clf.predict(X_test[i], node))
            prediction.append(pred)
    else:
        print('\nPPR')
        print(utilities.feature_models[model])
        similarity_m = simp_data[utilities.feature_models[model]]['S']
        prediction = ppr.predict(similarity_m, test_array)
    
    
    
    # fpr, fnr = accuracy_metrics.metrics(test_labels, prediction)
    count = 0
    total = 0
    for i in range(len(prediction)):
        print(prediction[i], Y_test[i])
        if Y_test[i] == prediction[i]:
            count +=1
        total +=1 
    print(count, total)
    
    
if __name__ == '__main__':
    start_task2() 