import os
import pickle
import cv2
import aggregation
import utilities


def start_task0():
    while True:
        input_folder_name = input('Enter folder to be stored or q if all folders are done:')
        if input_folder_name == 'q':
            break
        input_folder_path = os.path.join(os.getcwd(), input_folder_name)

        metadata_file = input_folder_name + '_metadata.pickle'
        simp_file = input_folder_name + '_simp.pickle'

        if not os.path.isdir(input_folder_path):
            print('Dataset not present: Please download image dataset, save dataset in folder "sample_images"')
            print("Terminating...")
            quit()

        if os.path.isfile(metadata_file) and os.path.isfile(simp_file):
            continue

        # Iterating through folder and finding image models
        if os.path.isfile(metadata_file):
            with open(metadata_file, 'rb') as handle:
                images = pickle.load(handle)
        else:
            images = {}
            for filename in os.listdir(input_folder_path):
                img = cv2.imread(os.path.join(input_folder_path, filename), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    image = {'image': img}
                    for i in range(3):
                        image[utilities.feature_models[i]] = utilities.feature_extraction[i](img)
                    x, y, z = filename.split('.')[0].split('-')[1:]
                    image['x_label'] = utilities.label_dict[x]
                    image['y_label'] = int(y)
                    image['z_label'] = int(z)
                    images[filename] = image
            # Saving image models in pickle file 'metadata.pickle
            with open(metadata_file, 'wb') as handle:
                pickle.dump(images, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Computing similarity & transformation matrices
        similarity_map = {}
        for i in range(3):
            types, type_matrix = aggregation.group_by_type_all(images, i)
            subjects, subject_matrix = aggregation.group_by_subject_all(images, i)
            image_ids, image_matrix = aggregation.group_by_image_all(images, i)
            similarity_map[utilities.feature_models[i]] = {
                'T': type_matrix,
                'S': subject_matrix,
                'I': image_matrix,
                'types': types,
                'subjects': subjects,
                'images': image_ids
            }

        # Saving similarity & transformation matrices in 'simp.pickle'
        with open(simp_file, 'wb') as handle:
            pickle.dump(similarity_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Task0 Successfully Executed, 2 dumps saved: %s %s' % (metadata_file, simp_file))


if __name__ == '__main__':
    start_task0()
