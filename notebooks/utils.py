from keras.preprocessing import image
from glob import glob
import numpy as np
import pandas as pd
import os


DATA_DIR = os.path.join('../data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train/*')
TRAIN_TARGETS = os.path.join(DATA_DIR, 'train_labels.csv')

def load_img(filepath):
    img = image.load_img(filepath)
    return image.img_to_array(img)

def load_data(filepath):
    files = glob(filepath)
    data = [load_img(img) for img in files]
    return np.array(data)

def load_train_data():
    files = glob(TRAIN_DIR)
    targets = dict(pd.read_csv(TRAIN_TARGETS).as_matrix())
    data = []
    y_true = []
    for img in files:
        filename = int(img.split('/')[-1].split('.')[0])
        data.append(load_img(img))
        y_true.append(targets[filename])
    return np.array(data), np.array(y_true)
