import os
import copyUtils.copyDirectoryUtils as copy_utils
from sklearn import metrics
import joblib
import params as params
import modelUtils.modelUtils as modelUtils
import numpy as np
import utils.utils as utils


use_base_model = params.testKMeans_use_base_model
test_on_train_data = False # If we test kmeans model on data that the model was trained this value should be true. It will work much faster

models_directory = params.models_directory
predicted_clusters_directory = params.predicted_clusters_directory
pca_path = os.path.join(models_directory, params.pca_model_name).__str__()
kmeans_path = os.path.join(models_directory, params.k_means_model_name).__str__()
model = None
if use_base_model is True:
    model = modelUtils.get_model(architecture=params.model_architecture, layer=params.model_layer)
else:
    model = modelUtils.get_model_witch_weights(params.model_architecture, params.model_layer, params.CNN_weights_path)

pca = joblib.load(pca_path)
k_means = joblib.load(kmeans_path)
images_shape_x = params.images_shape_x
images_shape_y = params.images_shape_y


test_dataset_directory = params.test_dataset_directory
dirs = os.listdir(test_dataset_directory)

test_images_array = []
test_features = []
test_images_category_array = []
test_features_selected = []
if test_on_train_data is False:
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
            test_images_array.append(path_to_file)
            test_images_category_array.append(dir.__str__())
            img = copy_utils.get_file(path_to_file, (images_shape_x, images_shape_y))
            features = modelUtils.get_output(img, model, architecture=params.model_architecture)
            test_features.append(features)

if test_on_train_data is True:
    features_path = params.extracted_features_directory
    features_array = os.listdir(features_path)
    for features in features_array:
        if copy_utils.is_system_file(features):
            continue
        features_path = os.path.join(params.extracted_features_directory, features).__str__()
        local_test_features, local_test_images_category_array, local_test_images_array = joblib.load(features_path)

        if len(test_images_array) == 0:
            test_images_array = local_test_images_array
        else:
            test_images_array = np.concatenate((test_images_array, local_test_images_array), axis=0)
        if len(test_features_selected) == 0:
            test_features_selected = local_test_features
        else:
            test_features_selected = np.concatenate((test_features_selected, local_test_features), axis=0)
        if len(test_images_category_array) == 0:
            test_images_category_array = local_test_images_category_array
        else:
            test_images_category_array = np.concatenate((test_images_category_array, local_test_images_category_array), axis=0)

    test_images_array = np.squeeze(np.array(test_images_array))
    test_features_selected = np.squeeze(np.array(test_features_selected))
    test_images_category_array = np.squeeze(np.array(test_images_category_array))

if test_on_train_data is False:
    test_features = np.array(test_features) # array with features vectors
    test_features_selected = pca.transform(test_features) # array with selected features vectors
test_predict = k_means.predict(test_features_selected) # predicted clusters
dist_centers = k_means.transform(test_features_selected) # distances from centers

if params.verbose is True:
    print(test_images_category_array)
    print(test_predict)
    print(dist_centers)

print("[-1, 1] (Best 1) Adjusted Rand index: ", metrics.adjusted_rand_score(test_images_category_array, test_predict))
print("[-1, 1] (Best 1) Mutual Information based scores: ", metrics.adjusted_mutual_info_score(test_images_category_array, test_predict))
print("[-1, 1] (Best 1) V-measure: ", metrics.v_measure_score(test_images_category_array, test_predict))
print("[-1, 1] (Best 1) Silhouette Coefficient: ", metrics.silhouette_score(test_features_selected, test_predict, metric='euclidean'))

copy_utils.save_predicted_clusters(predicted_clusters_directory, test_predict, test_images_array, dist_centers, True)
numeric_categories = utils.convert_to_numeric_labels(test_images_category_array)
utils.save_confusion_matrix(numeric_categories, test_predict, predicted_clusters_directory)

if params.verbose is True:
    print("Comparision")
    print(np.array(numeric_categories)) # real
    print(test_predict) # predicted


