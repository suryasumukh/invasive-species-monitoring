from glob import glob
import numpy as np
import os

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16

from tqdm import tqdm

DATA_DIR = os.path.join('../data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train/*')
TRAIN_TARGET = os.path.join(DATA_DIR, 'train_labels.csv')
TEST_DIR = os.path.join(DATA_DIR, 'test/*')


class TrainDataGenerator(object):
    """
    A Wrapper class to load hydrangea train data
    """
    TRAIN_DATA = os.path.join(DATA_DIR, 'train_data_features.npz')

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.train_data = np.load(self.TRAIN_DATA)
        self.nb_samples = len(self.train_data['labels'])

    def next_batch(self):
        """
        prepares training batches
        :return: returns images, y_true pairs of batch_size
        """
        _idx = 0
        while True:
            if _idx >= self.nb_samples:
                _idx = 0
            batch_imgs = self.train_data['features'][_idx: _idx+self.batch_size]
            batch_targets = self.train_data['labels'][_idx: _idx+self.batch_size]

            _idx += self.batch_size
            yield batch_imgs, batch_targets


class TestDataGenerator(object):
    """
    A wrapper to load hydrangea test data
    """
    TEST_DATA = os.path.join(DATA_DIR, 'test_data.npy')

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.test_data = np.load(self.TEST_DATA)
        self.nb_samples = self.test_data.shape[0]

    def next_batch(self):
        """
        prepares test batches
        :return: returns images, y_true pairs of batch_size
        """
        _idx = 0
        while True:
            if _idx >= self.nb_samples:
                _idx = 0
            batch_imgs = self.test_data[_idx: _idx+self.batch_size]
            _idx += self.batch_size
            yield batch_imgs


def load_train_data():
    targets = {}
    with open(TRAIN_TARGET, 'r') as file_:
        for line in file_:
            _img, _prob = line.split(',')
            targets[_img] = _prob

    train_imgs = []
    y_true = []

    for train_img in tqdm(glob(TRAIN_DIR)):
        img = image.load_img(train_img, target_size=(224, 224))
        pixels = image.img_to_array(img=img)
        train_imgs.append(np.expand_dims(pixels, axis=0))

        img_name = train_img.split('/')[-1].split('.')[0]
        y_true.append(targets[img_name])
    return np.vstack(train_imgs), np.array(y_true)


def load_test_data():
    test_imgs = []

    for test_img in tqdm(glob(TEST_DIR)):
        img = image.load_img(test_img, target_size=(224, 224))
        pixels = image.img_to_array(img=img)
        test_imgs.append(np.expand_dims(pixels, axis=0))

    return np.vstack(test_imgs)


def bottleneck_features(tensor):
    tensor = tensor.astype(float)/255
    vgg16 = VGG16(include_top=False, weights='imagenet')
    return vgg16.predict(preprocess_input(tensor))
