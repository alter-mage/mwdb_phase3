import os
import cv2
import task4
import task5
import numpy as np
import matplotlib.pyplot as plt

from svm import MulticlassSVM as SVM

index_map = [task4.get_top_images, task5.get_top_images]


# Function used to plot images
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


# Function used to obtain relevance feedback from the user.
def get_feedback():
    label_input = [int(i) for i in input(
        "enter comma separated 1 or 0 for each result, 1 for relevant, 0 for irrelevant"
    ).split(',')]
    return label_input


# Obtain trained SVM classifier based on input training data
def get_trained_model(features, labels):
    clf = SVM()
    clf.fit(np.array(features), labels)
    return clf


# Main function of task7
def start_task7():
    similar_images_task = int(input('Enter task based on which similar images required, 0 for LSH and 1 for VA-file:'))
    if similar_images_task not in [0, 1]:
        print('Similar images can only be retrieved based on task 4/5, incorrect input')
        print("Terminating...")
        return

    input_list, similarity_image_map, _, __ = index_map[similar_images_task]()
    t, query_image_name = input_list[-3], input_list[-1]
    query_image_path = os.path.join(os.getcwd(), query_image_name + '.png')
    query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)

    top_images = [image[2] for image in similarity_image_map[:t]]
    plot_results(top_images, query_image, t)

    top_k_train = []
    for i in similarity_image_map[:t]:
        top_k_train.append(i[1])

    labels = get_feedback()
    while not (len(labels) == t):
        labels = get_feedback()

    clf = get_trained_model(top_k_train, labels)

    test_features = [similarity_image[1] for similarity_image in similarity_image_map]
    prediction_results = clf.predict(test_features)

    relevant_images, counter = [], 2
    while len(relevant_images) < t:
        input_list[-3] *= counter
        input_list_temp, similarity_image_map_post_feedback, _, index_flag = index_map[similar_images_task](input_list)

        relevant_images, irrelevant_images = [], []
        for i in similarity_image_map_post_feedback:
            pred = clf.predict(i[1])
            if pred == 1:
                relevant_images.append(i)
            else:
                irrelevant_images.append(i)
        counter += 1

        if index_flag:
            relevant_images += irrelevant_images
            relevant_images = relevant_images[:t]

    plot_results([image[2] for image in relevant_images], query_image, t)


if __name__ == '__main__':
    start_task7()