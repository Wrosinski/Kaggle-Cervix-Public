import numpy as np
import os
import glob
import cv2
import pandas as pd
import time
import scipy
import warnings
warnings.filterwarnings("ignore")
from scipy import misc


def get_im_cv2(path):
    img = misc.imread(path, mode = 'RGB')
    img = misc.imresize(img, (299, 299))
    return img

def load_ids(src):
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
        for fl in files:
            flbase = fld + '/' + os.path.basename(fl)
            X_train_id.append(flbase)
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train_id

def load_ids_test(src):
    print('Read test images')
    start_time = time.time()
    files = sorted(glob.glob(src + '*.jpg'))
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        X_test_id.append(flbase)
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    X_test_id = np.array(X_test_id)
    return X_test_id 

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
        for fl in files:
            flbase = fld + '/' + os.path.basename(fl)
            try:
                img = get_im_cv2(fl)
            except ValueError:
                print('Failed to load:', fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    X_train = np.array(X_train)
    X_train = X_train.astype('float32')
    y_train = np.array(y_train)
    return X_train, y_train, X_train_id


def load_test(src):
    print('Read test images')
    start_time = time.time()
    files = sorted(glob.glob(src + '*.jpg'))
    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl)
        X_test.append(img)
        X_test_id.append(flbase)
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    X_test = np.array(X_test)
    X_test_id = np.array(X_test_id)
    return X_test, X_test_id 
