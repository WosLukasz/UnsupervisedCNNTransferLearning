from keras.models import load_model, Model
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as xception_preprocess_input
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from keras.applications.nasnet import NASNetLarge
from keras.applications.nasnet import preprocess_input as nasnet_preprocess_input
import numpy as np
import params as params
import tensorflow as tf


def get_model(architecture='Xception', layer='avg_pool'):
    """
    If there will be used other architecture than Xception, then there is a need to change output layer
    """
    base_model = None
    if architecture == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=True)
    elif architecture == 'Xception':
        base_model = Xception(weights='imagenet', include_top=True)
    elif architecture == 'VGG19':
        base_model = VGG19(weights='imagenet', include_top=True)
    elif architecture == 'NASNetLarge':
        base_model = NASNetLarge(weights='imagenet', include_top=True)

    if params.verbose is True:
        base_model.summary()

    output = Dense(params.clusters, activation=params.last_layer_activation)(base_model.layers[-2].output)
    new_model = Model(inputs=base_model.input, outputs=output)
    new_model.compile(optimizer=params.optimizer, loss=params.loss, metrics=params.metrics)

    if params.verbose is True:
        new_model.summary()

    return new_model


def get_model_witch_weights(architecture, layer, path):
    model = get_model(architecture=architecture, layer=layer)
    model.load_weights(path)

    return model


def get_model_by_path(path):
    return tf.keras.models.load_model(path)


def get_output(img_arr, base_model, architecture='Xception'):
    if img_arr.shape[2] == 1:
        img_arr = img_arr.repeat(3, axis=2)

    # (1, 224, 224, 3)
    arr4d = np.expand_dims(img_arr, axis=0)
    if architecture == 'VGG16':
        arr4d_pp = vgg16_preprocess_input(arr4d)
    elif architecture == 'Xception':
        arr4d_pp = xception_preprocess_input(arr4d)
    elif architecture == 'VGG19':
        arr4d_pp = vgg19_preprocess_input(arr4d)
    elif architecture == 'NASNetLarge':
        arr4d_pp = nasnet_preprocess_input(arr4d)

    model = Model(inputs=base_model.input, outputs=base_model.get_layer(params.model_layer).output)

    return model.predict(arr4d_pp)[0,:]


def prepare_data(img_arr, architecture='Xception'):
    if img_arr.shape[2] == 1:
        img_arr = img_arr.repeat(3, axis=2)

    # (1, 224, 224, 3)
    arr4d = np.expand_dims(img_arr, axis=0)
    if architecture == 'VGG16':
        arr4d_pp = vgg16_preprocess_input(arr4d)
    elif architecture == 'Xception':
        arr4d_pp = xception_preprocess_input(arr4d)
    elif architecture == 'VGG19':
        arr4d_pp = vgg19_preprocess_input(arr4d)
    elif architecture == 'NASNetLarge':
        arr4d_pp = nasnet_preprocess_input(arr4d)

    return arr4d_pp


def feature_vectors(imgs_dict, model, architecture='Xception'):
    f_vect = {}
    for fn, img in imgs_dict.items():
        f_vect[fn] = get_output(img, model, architecture)
    return f_vect
