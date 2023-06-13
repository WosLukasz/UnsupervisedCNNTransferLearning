import os
import copyUtils.copyDirectoryUtils as copy_utils
from sklearn import metrics
import joblib
import params as params
import numpy as np
import utils.utils as utils
import glob
from pathlib import Path

models_directory = params.models_directory
predicted_clusters_directory = params.predicted_clusters_directory
kmeans_path = os.path.join(models_directory, params.k_means_model_name).__str__()
k_means = joblib.load(kmeans_path)

test_dataset_directory = params.test_dataset_directory
dirs = os.listdir(test_dataset_directory)

test_images_array = []
test_features = []
test_images_category_array = []

for dir in dirs:
    if copy_utils.is_system_file(dir):
        continue
    print("Getting labels for class " + dir.__str__())
    source_dir_path = os.path.join(test_dataset_directory, dir).__str__()
    test_directory_file_names = copy_utils.get_file_names(source_dir_path)

    test_directory_path = os.path.join(source_dir_path, params.test_on_data).__str__()
    test_directory_file_names = copy_utils.get_file_names(test_directory_path)

    for file in test_directory_file_names:
        path_to_file = os.path.join(test_directory_path, file).__str__()
        test_images_array.append(path_to_file)
        test_images_category_array.append(dir.__str__())

print(np.shape(test_images_array))
print(np.shape(test_images_category_array))

pkls_direcotry = params.extracted_features_directory
pkls = sorted(Path(pkls_direcotry).iterdir(), key=os.path.getmtime)
print(pkls)

test_features = []
for pkl in pkls:
    print("Getting pkl " + pkl.__str__())
    local_features = joblib.load(pkl)
    if len(test_features) == 0:
        test_features = local_features
    else:
        test_features = np.concatenate((test_features, local_features), axis=0)


print(np.shape(test_features))
print(test_features)
test_features = np.array(test_features) # array with features vectors
test_predict = k_means.predict(test_features) # predicted clusters
dist_centers = k_means.transform(test_features) # distances from centers

if params.verbose is True:
    print(test_images_category_array)
    print(test_predict)
    print(dist_centers)

print("[-1, 1] (Best 1) Adjusted Rand index: ", metrics.adjusted_rand_score(test_images_category_array, test_predict))
print("[-1, 1] (Best 1) Mutual Information based scores: ", metrics.adjusted_mutual_info_score(test_images_category_array, test_predict))
print("[-1, 1] (Best 1) V-measure: ", metrics.v_measure_score(test_images_category_array, test_predict))
# print("[-1, 1] (Best 1) Silhouette Coefficient: ", metrics.silhouette_score(test_features, test_predict, metric='euclidean'))

# copy_utils.save_predicted_clusters(predicted_clusters_directory, test_predict, test_images_array, dist_centers, True)
# numeric_categories = utils.convert_to_numeric_labels(test_images_category_array)
# utils.save_confusion_matrix(numeric_categories, test_predict, predicted_clusters_directory)

if params.verbose is True:
    print("Comparision")
    print(np.array(numeric_categories)) # real
    print(test_predict) # predicted

