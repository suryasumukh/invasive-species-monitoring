from glob import glob
import numpy as np
import os

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


DATA_DIR = os.path.join('../data')


class TrainDataGenerator(object):
    """
    A Wrapper class to load hydrangea train data
    """
    TRAIN_DIR = os.path.join(DATA_DIR, 'train/*')
    TRAIN_TARGET = os.path.join(DATA_DIR, 'train_labels.csv')

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.train_imgs = glob(self.TRAIN_DIR)
        self.nb_samples = len(self.train_imgs)
        self.target_dict = self._load_targets()

    def _load_targets(self):
        """
        load train prediction probabilities
        :return: dict of filename, probability pairs
        """
        targets = {}
        with open(self.TRAIN_TARGET, 'r') as file_:
            for line in file_:
                _img, _prob = line.split(',')
                targets[_img] = _prob
        return targets

    def next_batch(self):
        """
        prepares training batches
        :return: returns images, y_true pairs of batch_size
        """
        _idx = 0
        while True:
            if _idx >= self.nb_samples:
                _idx = 0
            _batch_imgs = self.train_imgs[_idx: _idx+self.batch_size]

            batch_imgs = []
            batch_targets = []

            for img_path in _batch_imgs:
                train_img = image.load_img(img_path)
                train_pixels = image.img_to_array(train_img)
                train_pixels = np.expand_dims(train_pixels, axis=0)

                img_name = img_path.split('/')[-1].split('.')[0]
                probability = self.target_dict[img_name]

                batch_imgs.append(train_pixels)
                batch_targets.append(probability)

            _idx += self.batch_size
            batch_imgs = np.vstack(batch_imgs).astype('float32')/255
            batch_imgs = preprocess_input(batch_imgs)
            yield batch_imgs, np.array(batch_targets)


class TestDataGenerator(object):
    """
    A wrapper to load hydrangea test data
    """
    TEST_DIR = os.path.join(DATA_DIR, 'test/*')

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.test_imgs = glob(self.TEST_DIR)
        self.nb_samples = len(self.test_imgs)

    def next_batch(self):
        """
        prepares test batches
        :return: returns images, y_true pairs of batch_size
        """
        _idx = 0
        while True:
            if _idx >= self.nb_samples:
                _idx = 0
            _batch_imgs = self.test_imgs[_idx: _idx+self.batch_size]

            batch_imgs = []

            for img_path in _batch_imgs:
                test_img = image.load_img(img_path)
                test_pixels = image.img_to_array(test_img)
                test_pixels = np.expand_dims(test_pixels, axis=0).astype('float32')/255
                batch_imgs.append(test_pixels)

            _idx += self.batch_size
            batch_imgs = np.vstack(batch_imgs)
            yield preprocess_input(batch_imgs)
