import numpy as np
from keras.utils import Sequence
import copyUtils.copyDirectoryUtils as copy_utils
import params as params
import modelUtils.modelUtils as modelUtils


class My_Custom_Generator(Sequence):

    def __init__(self, image_filepaths, labels, batch_size):
        self.image_filepaths = image_filepaths
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_filepaths) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filepaths[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        imgs_to_return = np.array([
            modelUtils.prepare_data(
                copy_utils.get_file(file_path, (params.images_shape_x, params.images_shape_y))
                , architecture=params.model_architecture)
            for file_path in batch_x])

        imgs_to_return = np.squeeze(imgs_to_return)
        if len(imgs_to_return.shape) == 3:
            imgs_to_return = np.expand_dims(imgs_to_return, axis=0)

        return imgs_to_return, np.array(batch_y)