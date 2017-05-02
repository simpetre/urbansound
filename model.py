from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import scipy
from keras.models import Sequential
from keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelBinarizer

from PIL import Image
import numpy as np


import os

def feature_label_paths(path):
    """Crawl through directory of spectrograms to return absolute filenames
    INPUT: path (string): top directory to start crawl
    OUTPUT: feature_paths, targets (list of strings): Lists of spectrogram
    paths and targets for those spectrograms, respectively
    """ 
    feature_paths = []
    target_paths = []
    for label in labels:
        for filename in os.listdir(path + '/spectrograms/' + label):
            if filename[-3:] == 'png':
                feature_paths.append(path + '/spectrograms/' + label +
                                            '/' + filename)
                target_paths.append(label)
    return feature_paths, targets

def spectrogram_to_array(feature_paths, targets):
    """Converts spectrograms on disk to numpy arrays/labels for processing
    INPUT: feature_paths (list of strings): list of paths to feature spectrograms
    targets (list of strings): list of matching targets for spectrograms
    OUTPUT: feature_matrix (numpy array): matrix representation of spectrograms
    target_matrix (numpy array): one hot encoding of labels
    """

    features = []
    lb = LabelBinarizer()

    for filename in filenames:
        im = Image.open(filename)
        im_array = img_to_array(im)
        features.append(scipy.misc.imresize(im_array, size=(160, 120)))
    feature_matrix = np.array(features)
    target_matrix = lb.fit_transform(targets)
    return feature_matrix, target_matrix

def specify_model():
    """Specify layer types/parameters for convolutional neural network
    OUTPUT: model (Keras object): CNN model
    """
    model = Sequential()
    model.add(Convolution2D(16, 5, input_shape=(160, 120, 4)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 5))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(256, 5))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('tanh'))

    model.add(Dense(256))
    model.add(Activation('tanh'))

    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('sigmoid'))
    return model

if __name__ == '__main__':
    feature_paths, targets = feature_label_paths()
    features, targets = spectrogram_to_array(feature_paths, targets)
    model = specify_model()
    X_train, X_test, y_train, y_test = train_test_split(features, targets)
    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    model.fit(X_train, y_train, callbacks=[earlystopping])
    y_pred = model.predict(X_test)

    accuracy_score(y_pred, y_test)
