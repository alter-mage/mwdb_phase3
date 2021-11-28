import task4
from decision_tree import DecisionTreeClassifier
import numpy as np
import os
from task4 import get_top_images
import cv2
import matplotlib.pyplot as plt
import utilities


def plot_results(top_images, query_image, t):
    fig, axes = plt.subplots(t + 1, 1)
    for i, axis in enumerate(axes):
        if i == 0:
            img = query_image
            # axis.text(74, 25, query, size=9)
            axis.text(74, 45, 'Original image', size=9)
        else:
            img = top_images[i - 1]
        axis.imshow(img, cmap='gray')
        axis.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    fig.suptitle(str(t) + ' most similar images')
    plt.show()


def get_feedback():
    label_input = [int(i) for i in input(
        "enter comma separated 1 or 0 for each result, 1 for relevant, 0 for irrelevant"
    ).split(',')]
    return label_input


def get_predictor_object(top_k_relevant, labels_for_top_k):

    X_train = top_k_relevant
    #before this you have to change the label from 1,0 to +1 and -1
    y_train = labels_for_top_k

    clf = DecisionTreeClassifier()
    node = clf.fit(X_train, y_train)
    
    return clf, node

    
def start_task6():
    l = int(input('enter num of layers: '))
    k = int(input('enter num of hashes per layer: '))
    vector_file = input('enter vector file: ')
    t = int(input('enter number of retrievals: '))
    image_folder = os.path.join(os.getcwd(), input('enter image folder: '))
    query_image_path = os.path.join(os.getcwd(), input('enter query image name') + '.png')
    query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
    
    similarity_image_map, _, __ = get_top_images(
        l, k, vector_file, t, image_folder, query_image
    )

    top_images = [image[2] for image in similarity_image_map[:t]]
    plot_results(top_images, query_image, t)

    top_k_train = []
    for i in top_images:
        top_k_train.append(i[1])

    labels = get_feedback()
    while not (len(labels) == t):
        labels = get_feedback()

    clf, node = get_predictor_object(top_k_train, labels)

    relevant_images, counter = [], 2
    while len(relevant_images) < t:
        similarity_image_map_post_feedback, _, index_flag = task4.get_top_images(
            l, k, vector_file, counter*t, image_folder, query_image
        )

        relevant_images, irrelevant_images = [], []
        for i in similarity_image_map_post_feedback:
            pred = clf.predict(i[1], node)
            if pred == 1:
                relevant_images.append(i)
            else:
                irrelevant_images.append(i)
        counter += 1

        if index_flag:
            relevant_images += irrelevant_images
            relevant_images = relevant_images[:t]
    
    plot_results(relevant_images, query_image, t)
    print(len(relevant_images))
    print("done")


if __name__ == "__main__":
    start_task6()    
