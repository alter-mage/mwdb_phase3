from decision_tree import DecisionTreeClassifier
import numpy as np
import os
from task4 import get_top_images
import cv2
import matplotlib.pyplot as plt
import utilities


def get_feedback():
    label_input = [int(i) for i in input("enter 1 or 0").split()]
    return label_input


def get_predictor_object(top_k_relevant, labels_for_top_k):

    X_train = top_k_relevant
    #before this you have to change the label from 1,0 to +1 and -1
    y_train = labels_for_top_k
    training_data = []
    for i in range(len(X_train)):
        a =[] 
        a.extend(X_train[i])
        a.append(y_train[i])
        training_data.append(a)

    clf = DecisionTreeClassifier()
    node = clf.fit(training_data)
    
    return clf,node

    
def start_task6():
    l = int(input('enter num of layers: '))
    k = int(input('enter num of hashes per layer: '))
    vector_file = input('enter vector file: ')
    t = int(input('enter number of retrievals: '))
    image_folder = os.path.join(os.getcwd(), input('enter image folder: '))
    query_image_path = os.path.join(os.getcwd(), input('enter query image name') + '.png')
    query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
    
    similarity_image_map, vector_file_tokens = get_top_images(
        l, k, vector_file, t, image_folder, query_image
    )

    top_images = similarity_image_map[:t]
    fig, axes = plt.subplots(t + 1, 1)
    for i, axis in enumerate(axes):
        if i == 0:
            img = query_image
            # axis.text(74, 25, query, size=9)
            axis.text(74, 45, 'Original image', size=9)
        else:
            img = top_images[i - 1][2]
            axis.text(74, 45, str(top_images[i - 1][0]), size=9)
        axis.imshow(img, cmap='gray')
        axis.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    fig.suptitle(str(t) + ' most similar images - ' +
                 utilities.similarity_measures[utilities.feature_models.index(vector_file_tokens[1])], size=10)
    
    plt.show()

    #now you will have t images and t labels
    top_k_train = []
    for i in top_images:
        top_k_train.append(i[1])

    labels = get_feedback()
    while not (len(labels) == t):
        labels = get_feedback()

    clf,node = get_predictor_object(top_k_train , labels)
    
    result = []
    for i in similarity_image_map:
        pred = clf.predict(i[1],node)
        print(pred)
        if pred == 1:
            result.append(i)
    
    print(len(result))
    print("done")


        
if __name__ == "__main__":
    start_task6()    
