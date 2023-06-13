import os
import keras
import copyUtils.copyDirectoryUtils as copy_utils
import joblib
import params as params
import modelUtils.modelUtils as modelUtils
import numpy as np
import utils.utils as utils
from modelUtils.gpu import gpu_fix
import cv2
from sklearn import metrics

gpu_fix()

predicted_in_path = params.predicted_clusters_directory
results_path = params.predicted_clusters_directory
array_path = os.path.join(predicted_in_path, params.out_array_name).__str__()
smallest_num = 2 # always 2
images_shape_x = params.images_shape_x
images_shape_y = params.images_shape_y

model = modelUtils.get_model(architecture=params.model_architecture, layer=params.model_layer)

images_paths_array = []
labels = []

test_on_clean_data = True

if test_on_clean_data is True:
    test_dataset_directory = params.test_dataset_directory
    dirs = os.listdir(test_dataset_directory)
    for dir in dirs:
        if copy_utils.is_system_file(dir):
            continue
        print("Getting data paths from class " + dir.__str__())
        source_dir_path = os.path.join(test_dataset_directory, dir).__str__()
        test_directory_file_names = copy_utils.get_file_names(source_dir_path)

        test_directory_path = os.path.join(source_dir_path, params.test_on_data).__str__()
        test_directory_file_names = copy_utils.get_file_names(test_directory_path)

        for file in test_directory_file_names:
            path_to_file = os.path.join(test_directory_path, file).__str__()
            images_paths_array.append(path_to_file)
            labels.append(dir.__str__())
else:
    data_array = joblib.load(array_path)
    if params.verbose is True:
        print(data_array)
    for i in range(0, len(data_array)):
        path = data_array[i][0]
        img = cv2.imread(path)
        proper_path = path
        label = data_array[i][1]
        if img is None:
            continue
            # junk_path = copy_utils.get_junk_path(path)
            # junk_img = cv2.imread(junk_path)
            # if junk_img is not None:
            #     proper_path = junk_path
            #     if params.verbose is True:
            #         print("Bad label:" + proper_path)
            #     label = list(data_array[i][2]).index(sorted(data_array[i][2])[smallest_num - 1])
            # else:
            #     continue

        images_paths_array.append(proper_path)
        labels.append(label)

if params.verbose is True:
    print(images_paths_array)
    print(labels)


model = modelUtils.get_model_witch_weights(params.model_architecture, params.model_layer, params.CNN_weights_path)

batch_size = params.batchSize
rest = len(images_paths_array) % batch_size
epochs = int(len(images_paths_array) / batch_size)
if rest != 0:
    epochs = epochs + 1
predicted = []
paths = []
for i in range(epochs):
    data = []
    print("Batch [" + str(i) + "/" + str(epochs - 1) + "]")
    start = i * batch_size
    end = start + batch_size
    if len(images_paths_array) < end:
        end = len(images_paths_array)
    bath_images_paths = images_paths_array[start:end]
    bath_labels = labels[start:end]
    for j in range(len(bath_images_paths)):
        path = bath_images_paths[j]
        paths.append(path)
        img = copy_utils.get_file(path, (images_shape_x, images_shape_y))
        prepared_image = modelUtils.prepare_data(img, architecture=params.model_architecture)
        data.append(prepared_image)

    data = np.squeeze(np.array(data))
    bath_labels = np.squeeze(np.array(bath_labels))
    original_labels = bath_labels

    y0 = model.predict(data)
    classes = np.argmax(y0, axis = 1)

    if len(predicted) == 0:
        predicted = classes
    else:
        predicted = np.concatenate((predicted, classes), axis=0)

    if params.verbose is True:
        print("==============")
        print("Predicted: ")
        print(classes)
        print("True: ")
        print(original_labels)


copy_utils.save_predicted_clusters(params.predicted_classes_directory, predicted, paths, None, False)

if test_on_clean_data is True:
    labels = utils.convert_to_numeric_labels(labels)

print("[-1, 1] (Best 1) Adjusted Rand index: ", metrics.adjusted_rand_score(labels, predicted))
print("[-1, 1] (Best 1) Mutual Information based scores: ", metrics.adjusted_mutual_info_score(labels, predicted))
print("[-1, 1] (Best 1) V-measure: ", metrics.v_measure_score(labels, predicted))

utils.save_confusion_matrix(labels, predicted, params.predicted_classes_directory)