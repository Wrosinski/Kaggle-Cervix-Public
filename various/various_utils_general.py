import numpy as np
np.random.seed(2016)

import os
import glob
import cv2
import datetime
import pandas as pd
import time
import scipy
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras import __version__ as keras_version

size = (299,299)

def get_im_cv2(path):
    img = scipy.misc.imread(path, mode = 'RGB')
    img = scipy.misc.imresize(img, size)
    return img

def load_train(src):
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()
    print('Read train images')
    folders = ['Type_1', 'Type_2', 'Type_3']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join(src, fld, '*.jpg')
        files = glob.glob(path)
        for i, fl in enumerate(files):
            flbase = os.path.basename(fl)
            flbase = fld + '/' + flbase
            try:
                img = get_im_cv2(fl)
                X_train.append(img)
                X_train_id.append(flbase)
                y_train.append(index)
            except Exception:
                print('Failed for image:', fl)
                continue
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    y_train = np.array(y_train)
    X_train = np.array(X_train)
    X_train = X_train.astype('uint8')
    print(X_train.shape)
    return X_train, y_train, X_train_id

def load_test(src):
    print('Read test images')
    start_time = time.time()
    files = sorted(glob.glob(src + '*.jpg'))
    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        try:
            img = get_im_cv2(fl)
            X_test.append(img)
            X_test_id.append(flbase)
        except Exception:
            print('Failed for image:', fl)
            continue
    print('Read test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    X_test = np.array(X_test)
    X_test = X_test.astype('uint8')
    X_test_id = np.array(X_test_id)
    print(X_test.shape)
    return X_test, X_test_id

def load_full(src):
    X_train = []
    X_train_id = []
    start_time = time.time()
    print('Read train images')
    folders = ['Type_1', 'Type_2', 'Type_3', 'test_renamed']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join(src, fld, '*.jpg')
        files = glob.glob(path)
        for i, fl in enumerate(files):
            flbase = os.path.basename(fl)
            flbase = fld + '/' + flbase
            try:
                img = get_im_cv2(fl)
                X_train.append(img)
                X_train_id.append(flbase)
            except Exception:
                print('Failed for image:', fl)
                continue
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    X_train = np.array(X_train)
    X_train = X_train.astype('uint8')
    print(X_train.shape)
    return X_train, X_train_id


def resize_train(src, dst):
    start_time = time.time()
    print('Read train images')
    folders = ['Type_1', 'Type_2', 'Type_3']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join(src, fld, '*.jpg')
        files = glob.glob(path)
        for i, fl in enumerate(files):
            flbase = os.path.basename(fl)
            flbase = '{}/id{}_origtrain_{}'.format(fld, flbase.split('.')[0], i) + '.jpg'
            try:
                img = get_im_cv2(fl)
            except Exception:
                print('Failed for image:', fl)
                continue
            res_img = cv2.resize(img, size, cv2.INTER_AREA)
            final = Image.fromarray((res_img).astype(np.uint8))
            final.save(dst + flbase)
    print('Resized train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return

def resize_test(src, dst):
    print('Read test images')
    start_time = time.time()
    files = sorted(glob.glob(src + '*.jpg'))
    for fl in files:
        flbase = os.path.basename(fl)
        flbase = flbase.split('.')[0] + '_resized' + '.jpg'
        try:
            img = get_im_cv2(fl)
        except Exception:
            print('Failed for image:', fl)
            continue
        res_img = cv2.resize(img, size, cv2.INTER_AREA)
        final = Image.fromarray((res_img).astype(np.uint8))
        final.save(dst + flbase)
    print('Resized test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return

def get_labels_ids_train(src):
    X_train_id = []
    y_train = []
    start_time = time.time()
    print('Read train images')
    folders = ['Type_1', 'Type_2', 'Type_3']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join(src, fld, '*.jpg')
        files = glob.glob(path)
        for i, fl in enumerate(files):
            flbase = os.path.basename(fl)
            try:
                X_train_id.append(flbase)
                y_train.append(index)
            except Exception:
                print('Failed for image:', fl)
                continue
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    y_train = np.array(y_train)
    return y_train, X_train_id


def get_labels_ids_test(src):
    print('Read test images')
    start_time = time.time()
    files = sorted(glob.glob(src + '*.jpg'))
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        try:
            X_test_id.append(flbase)
        except Exception:
            print('Failed for image:', fl)
            continue
    print('Read test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    X_test_id = np.array(X_test_id)
    return X_test_id
