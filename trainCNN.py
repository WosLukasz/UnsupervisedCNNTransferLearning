import os
import keras
import joblib
import params as params
import modelUtils.modelUtils as modelUtils
import numpy as np
from modelUtils.gpu import gpu_fix
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from modelUtils.My_Custom_Generator import My_Custom_Generator
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import copyUtils.copyDirectoryUtils as copy_utils


def filter_files(path):
    file_names = os.listdir(path)
    return list(filter(lambda x: "." not in x, file_names))


use_base_model = params.trainCNN_use_base_model
next_iteration_training = True


gpu_fix()

predicted_in_path = params.predicted_clusters_directory
results_path = params.predicted_clusters_directory
array_path = os.path.join(predicted_in_path, params.out_array_name).__str__()
smallest_num = 2 # always 2
images_shape_x = params.images_shape_x
images_shape_y = params.images_shape_y


if use_base_model is True:
    model = modelUtils.get_model(architecture=params.model_architecture, layer=params.model_layer)
else:
    model = modelUtils.get_model_witch_weights(params.model_architecture, params.model_layer, params.CNN_weights_input_path)


print("[Start] Start collecting paths and labels...")
images_paths_array = []
labels = []

#images_paths_array, first_labels = copy_utils.get_images_list(params.images_main_directory)
#first_labels = utils.convert_to_numeric_labels(first_labels)
#labels = map_labels(first_labels)

if next_iteration_training is False:
    data_array = joblib.load(array_path)

    if params.verbose is True:
        print(data_array)
    for i in range(0, len(data_array)):
        path = data_array[i][0]
        img = cv2.imread(path)
        proper_path = path
        label = data_array[i][1]
        if img is None:
            continue # comment this
            # junk_path = copy_utils.get_junk_path(path)
            # junk_img = cv2.imread(junk_path)
            # if junk_img is not None:
            #     proper_path = junk_path
            #     print("Bad label:" + proper_path)
            #     label = list(data_array[i][2]).index(sorted(data_array[i][2])[smallest_num - 1])
            # else:
            #     continue

        images_paths_array.append(proper_path)
        labels.append(label)

if next_iteration_training is True:
    train_data_path = params.previous_predicted_clusters_directory
    dirs = filter_files(train_data_path)
    for dir in dirs:
        source_dir_path = os.path.join(train_data_path, dir).__str__()
        file_names = copy_utils.get_file_names(source_dir_path)
        for file in file_names:
            proper_path = os.path.join(source_dir_path, file).__str__()
            images_paths_array.append(proper_path)
            labels.append(dir)


print(np.shape(images_paths_array))
print(np.shape(labels))

if params.verbose is True:
    print(images_paths_array)
    print(labels)

print("[Stop] Stop collecting paths and labels...")

# convert to numpy and to one-hot
images_paths_array = np.squeeze(np.array(images_paths_array))
labels = np.squeeze(np.array(labels))
original_labels = labels
labels = keras.utils.to_categorical(labels, num_classes=params.clusters)

#shuffle
images_paths_array, labels = shuffle(images_paths_array, labels)

#split
X_train_paths, X_val_paths, y_train, y_val = train_test_split(images_paths_array, labels, test_size=0.2, random_state=1)

if params.verbose is True:
    print(X_train_paths.shape)
    print(y_train.shape)
    print(X_val_paths.shape)
    print(y_val.shape)

# Create generators
batch_size = params.batch_size

my_training_batch_generator = My_Custom_Generator(X_train_paths, y_train, batch_size)
my_validation_batch_generator = My_Custom_Generator(X_val_paths, y_val, batch_size)


early_stopping = EarlyStopping(monitor='val_loss', patience=7, verbose=0, mode='min')
checkpoints = ModelCheckpoint(params.CNN_weights_path, save_best_only=True, monitor='val_loss', mode='min')
# log_dir = os.path.join('logs')
# tensor_board = TensorBoard(log_dir=log_dir)

history = model.fit_generator(generator=my_training_batch_generator,
                    steps_per_epoch=int(len(X_train_paths) // batch_size),
                    epochs=params.epochs,
                    validation_data=my_validation_batch_generator,
                    validation_steps=int(len(X_val_paths) // batch_size),
                    callbacks=[early_stopping, checkpoints]) #, tensor_board

model.save(params.CNN_model_path)
model.save_weights(params.CNN_weights_path)

if params.verbose is True:
    acc = history.history['acc']
    mae = history.history['mae']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(history.history['val_loss']))
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Categorical Accuracy')
    plt.plot(epochs_range, mae, label='Validation Categorical Accuracy')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Categorical Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()