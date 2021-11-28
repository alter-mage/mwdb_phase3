import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from task4 import get_top_images
from svm import MulticlassSVM as SVM


def get_feedback(t):
    print("In the order of display, please enter if image is relevant (1) or irrelevant (0)")
    relevances = []
    for index in range(t):
        relevance = int(input("Relevant (1) or irrelevant (0): "))
        relevances.append(relevance)
    return relevances


def plot_images(t, query_image, top_images):
    fig, axes = plt.subplots(t + 1, 1)
    for i, axis in enumerate(axes):
        if i == len(top_images):
            break
        if i == 0:
            img = query_image
            # axis.text(74, 25, query, size=9)
            axis.text(74, 45, 'Original image', size=9)
        else:
            img = top_images[i - 1][2]
            axis.text(74, 45, str(top_images[i - 1][0]), size=9)
        axis.imshow(img, cmap='gray')
        axis.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    # fig.suptitle(str(t) + ' most similar images - ' +
    #              utilities.similarity_measures[utilities.feature_models.index(vector_file_tokens[1])], size=10)
    plt.show()


def get_trained_model(features, labels):
    clf = SVM()
    clf.fit(np.array(features), labels)
    return clf


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

    similarity_image_map, vector_file_tokens = get_top_images(l, k, vector_file, t, image_folder, query_image)

    top_images = similarity_image_map[:t]
    plot_images(t, query_image, top_images)
    labels = get_feedback(t)

    features = [image[1] for image in top_images]
    labels = [-1 if label == 0 else 1 for label in labels]
    clf = get_trained_model(features, labels)

    test_features = [similarity_image[1] for similarity_image in similarity_image_map]
    prediction_results = clf.predict(test_features)

    image_with_scores = []
    for index in range(0, len(prediction_results)):
        prediction_result = prediction_results[index]
        if prediction_result == 1:
            image_with_scores.append(similarity_image_map[index])

    plot_images(t, query_image, image_with_scores)




if __name__ == '__main__':
    start_task7()