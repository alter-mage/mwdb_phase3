import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def fit(data_matrix, label_matrix):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='poly', degree=5))
    clf.fit(data_matrix, label_matrix)
    return clf