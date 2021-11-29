import os
import cv2
import task4
import numpy as np
import matplotlib.pyplot as plt

from svm import MulticlassSVM as SVM


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
    similar_images_task = int(input('Enter task based on which similar images required [4/5]:'))
    if (similar_images_task != 4 and similar_images_task != 5):
        print('Similar images can only be retrieved based on task 4/5, incorrect input')
        print("Terminating...")
        quit()

    l = int(input('Enter the number of layer: '))
    k = int(input('Enter the number of hash per layer: '))
    vector_file = input('Input vector file: ')
    t = int(input('Enter the number of similar images required: '))
    image_folder = os.path.join(os.getcwd(), input('Enter image folder name: '))

    query_image_path = os.path.join(os.getcwd(), input('Enter query image name') + '.png')
    query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)

    similarity_image_map, _, __ = task4.get_top_images(l, k, vector_file, t, image_folder, query_image)

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
        similarity_image_map_post_feedback, _, index_flag = task4.get_top_images(
            l, k, vector_file, counter * t, image_folder, query_image
        )

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

    # for index in range(0, len(prediction_results)):
    #     prediction_result = prediction_results[index]
    #     if prediction_result == 1:
    #         image_with_scores.append(similarity_image_map[index])

    plot_results([image[2] for image in relevant_images], query_image, t)


if __name__ == '__main__':
    start_task7()