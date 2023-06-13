import os

# Augmentation params
augmentation_images_number = '130'  # number of images that should be after augmentation for single class
dim_min = 200  # scaling images while augmentation to process smaller images later
dim_max = 500  # scaling images while augmentation to process smaller images later

# Main directories params
main_dataset_directory = 'D:/final_datasets/'  # main directory with datasets
main_results_directory = 'D:/dataset/experiment_last/'  # main directory where will be all results
original_dataset_directory = os.path.join(main_dataset_directory, 'dataset').__str__()  # path to original dataset

images_main_directory = os.path.join(main_dataset_directory, 'dataset_augmented/').__str__()  # path to train data direcotory
test_dataset_directory = os.path.join(main_dataset_directory, 'dataset_augmented/').__str__()  # path to test data direcotory (in typical case same like images_main_directory)

# Training params
clusters = 5  # clusters number
batchSize = 100  # batchSize in k_means and PCA algorithms (batchSize > clusters && batchSize >= selected_features_number)
selected_features_number = 100  # used in PCA (selected_features_number >= clusters)
images_shape_x = 299  # x shape of image (VGG16/VGG19: 224, Xception: 299, NASNetLarge: 331)
images_shape_y = 299  # y shape of image (VGG16/VGG19: 224, Xception: 299, NASNetLarge: 331)
pca_model_name = "XceptionPca.model"  # name of pca model saved in models_directory
k_means_model_name ="XceptionKmeans.model"  # name of kmeans model saved in models_directory
out_array_name = "array.array" # name of helper array saved in predicted_clusters_directory
junk_directory = "junk_directory" # name of junk directory where will be images that not fit to other in clusters
verbose = False # information if we want to print extra informations or not

#CNN params and hyperparameters
model_architecture = "Xception"  # architecture of CNN used in project
model_layer = "avg_pool"  # layer of CNN which will be used as output layer (currently not used)
last_layer_activation = 'softmax'  # type of output layer to classification
optimizer = 'Adam'  # optimizer used in CNN training
loss = "categorical_crossentropy"  # loss function used in CNN training
metrics = ["mae", "acc"]  # metrics used in CNN training
batch_size = 8  # batch size used in CNN training (the bigger the better)
epochs = 8  # epochs used in CNN training


## SECTION STEP 0 ##
# # Directories params
extracted_features_directory = os.path.join(main_results_directory, 'features_extracted').__str__()  # path to direcotry where will be extracted features from pictures
predicted_clusters_directory = os.path.join(main_results_directory, 'predicted_kMeans').__str__()  # path to directory where will be clusters predicted by kmeans
models_directory = os.path.join(main_results_directory, 'models').__str__()  # path to direcotory with all models

CNN_model_path = os.path.join(main_results_directory, 'trained_cnn.model').__str__()  # path to trained CNN model (not used in project)
CNN_weights_input_path = os.path.join(main_results_directory, 'trained_cnn_weights.model').__str__()  # path to trained CNN model weights
CNN_weights_path = os.path.join(main_results_directory, 'trained_cnn_weights.model').__str__()  # path to trained CNN model weights
predicted_classes_directory = os.path.join(main_results_directory, 'predicted_CNN').__str__()  # oath to directory where will be clusters predicted by CNN


# Training params
test_on_data = "train"  # information on which dataset test model (values: train or test)
features_extraction_use_base_model = True
trainKMeans_use_base_model = True
testKMeans_use_base_model = True

## SECTION STEP 1 ##
# # Directories params
# previous_predicted_clusters_directory = os.path.join(main_results_directory, 'predicted_kMeans').__str__()
#
# extracted_features_directory = os.path.join(main_results_directory, 'features_extracted_01').__str__()  # path to direcotry where will be extracted features from pictures
# predicted_clusters_directory = os.path.join(main_results_directory, 'predicted_kMeans_01').__str__()  # path to directory where will be clusters predicted by kmeans
# models_directory = os.path.join(main_results_directory, 'models_01').__str__()  # path to direcotory with all models
#
# CNN_model_path = os.path.join(main_results_directory, 'trained_cnn.model').__str__()  # path to trained CNN model (not used in project)
# CNN_weights_input_path = os.path.join(main_results_directory, 'trained_cnn_weights.model').__str__()  # path to trained CNN model weights
# CNN_weights_path = os.path.join(main_results_directory, 'trained_cnn_weights.model').__str__()  # path to trained CNN model weights
# predicted_classes_directory = os.path.join(main_results_directory, 'predicted_CNN_01').__str__()  # oath to directory where will be clusters predicted by CNN
#
# # Training params
# test_on_data = "train"  # information on which dataset test model (values: train or test)
# features_extraction_use_base_model = False
# trainKMeans_use_base_model = False
# testKMeans_use_base_model = False
# trainCNN_use_base_model = True

## SECTION STEP 2 ##
# # Directories params
# previous_predicted_clusters_directory = os.path.join(main_results_directory, 'predicted_CNN_01').__str__()
#
# extracted_features_directory = os.path.join(main_results_directory, 'features_extracted_02').__str__()  # path to direcotry where will be extracted features from pictures
# predicted_clusters_directory = os.path.join(main_results_directory, 'predicted_kMeans_02').__str__()  # path to directory where will be clusters predicted by kmeans
# models_directory = os.path.join(main_results_directory, 'models_02').__str__()  # path to direcotory with all models
#
# CNN_model_path = os.path.join(main_results_directory, 'trained_cnn_02.model').__str__()  # path to trained CNN model (not used in project)
# CNN_weights_input_path = os.path.join(main_results_directory, 'trained_cnn_weights.model').__str__()  # path to trained CNN model weights
# CNN_weights_path = os.path.join(main_results_directory, 'trained_cnn_weights_02.model').__str__()  # path to trained CNN model weights
# predicted_classes_directory = os.path.join(main_results_directory, 'predicted_CNN_02').__str__()  # oath to directory where will be clusters predicted by CNN
#
# # Training params
# test_on_data = "test"  # information on which dataset test model (values: train or test)
# features_extraction_use_base_model = False
# trainKMeans_use_base_model = False
# testKMeans_use_base_model = False
# trainCNN_use_base_model = False