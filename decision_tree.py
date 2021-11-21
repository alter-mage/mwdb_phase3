from sklearn import tree

def fit(Data_matrix, label_matrix):
    clf = tree.DecisionTreeClassifier()
    clf.fit(Data_matrix,label_matrix)
    return clf