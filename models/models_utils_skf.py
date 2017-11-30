import gc
import glob
import cv2
import datetime
import time
import scipy
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import tensorflow as tf

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import log_loss

from models_utils_loading import *
from models_utils_train import *
np.random.seed(1337)
tf.set_random_seed(1337)

epochs_number = 200
img_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.1,
                zoom_range=0.25,
                rotation_range=45,
                width_shift_range=0.25,
                height_shift_range=0.25,
                horizontal_flip=True,
                channel_shift_range=0.07)

checks_src = '/media/w/1c392724-ecf3-4615-8f3c-79368ec36380/DS Projects/Kaggle/Intel_Cervix/scripts/models/checks/'
oof_src = '/media/w/1c392724-ecf3-4615-8f3c-79368ec36380/DS Projects/Kaggle/Intel_Cervix/scripts/models/OOF_preds/'

# Model Training
def skf_model_run(X, y, X_test, train_ids, num_folds,
              stratify = True, modelname = None, savename = None, load_name = None):
    
    train_datagen = img_datagen
    train_datagen.fit(X, augment = True)
    valid_datagen = ImageDataGenerator(rescale=1./255,)
    if 'resnet' or 'inception' in savename:
        batch_size = 16
    if 'xception' in savename:
        batch_size = 8
    if 'dense' in savename:
        stop_patience = 8
    else:
        stop_patience = 5
        
    losses = []
    min_losses = []
    loss_history = []
    oof_train = np.zeros((X.shape[0], 3))
    oof_test = np.zeros((num_folds, X_test.shape[0], 3))
    
    os.makedirs('{}{}'.format(checks_src, savename), exist_ok = True)
    if stratify:
        print('Running Stratified K-Fold split.')
        folds_train_imgs, folds_val_imgs, folds_train_inds, folds_val_inds = split_proper_skf(train_ids, num_folds)
    else:
        print('Running standard K-Fold split.')
        folds_train_imgs, folds_val_imgs, folds_train_inds, folds_val_inds = split_proper_kf(train_ids, num_folds)
        
    for i in range(len(folds_train_inds)):
        print('Training on fold:', i + 1, '\n')
        X_tr = X[folds_train_inds[i]]
        X_val = X[folds_val_inds[i]]
        y_tr = y[folds_train_inds[i]]
        y_val = y[folds_val_inds[i]]
        
        callbacks = [ModelCheckpoint('{}{}/{}_fold{}.h5'.format(checks_src, savename, savename, i + 1), 
                                        monitor='val_loss', 
                                        verbose = 0, save_best_only = True),
                EarlyStopping(monitor='val_loss', patience = stop_patience, verbose = 1),
                CSVLogger('{}{}/{}_fold{}_history.csv'.format(checks_src, savename, savename, i + 1), append = True),
                ReduceLROnPlateau(monitor='val_loss', factor = 0.5, verbose = 1, 
                                  patience = 3, min_lr = 1e-5)]
        if load_name is not None:
            model = load_model(checks_src + load_name)
        else:
            model = modelname()

        history = model.fit_generator(train_datagen.flow(X_tr, y_tr, batch_size = batch_size), 
                            steps_per_epoch = X_tr.shape[0]/batch_size,
                            validation_data = valid_datagen.flow(X_val, y_val, batch_size = batch_size, shuffle = False),
                            validation_steps = X_val.shape[0]/batch_size, epochs = epochs_number, callbacks = callbacks)
        
        preds = model.predict(X_val/255., batch_size = 16)
        oof_train[folds_val_inds[i], :] = preds
        oof_test[i, :, :] = model.predict(X_test/255., batch_size = 16)
        loss = log_loss(y_val, preds)
        losses.append(loss)
        validation_loss = history.history['val_loss']
        loss_history.append(validation_loss)
        min_losses.append(np.min(validation_loss))
        print('Loss for last epoch for fold {}, : {}'.format(i + 1, loss))
        print('Minimum loss for current fold:', np.min(validation_loss), '\n',
              'Mean loss for current fold:', np.mean(validation_loss))
        
    oof_test = oof_test.mean(axis = 0)
    print('Test predictions shape:', oof_test.shape)
    np.save('{}train/{}'.format(oof_src, savename), oof_train)
    np.save('{}test/{}'.format(oof_src, savename), oof_test)
    
    print('Last epochs losses:', losses, '\n')
    print('Mean loss for model based on latest epoch:', np.array(losses).mean(axis = 0), '\n', '\n')
    print('Statistics for minimum losses per fold in current SKF run: \n',
          'Mean: {}'.format(np.mean(min_losses)), '\n',
          'Minimum: {}'.format(np.min(min_losses)), '\n',
          'Maximum: {}'.format(np.max(min_losses)), '\n',
          'Standard Deviation: {}'.format(np.std(min_losses)), '\n')
    with open('{}{}/{}_stats.txt'.format(checks_src, savename, savename), 'w') as text_file:
        text_file.write('Statistics for minimum losses per fold in current SKF run: \n')
        text_file.write('Minimum: {} \n'.format(np.min(min_losses)))
        text_file.write('Maximum: {} \n'.format(np.max(min_losses)))
        text_file.write('Mean: {} \n'.format(np.mean(min_losses)))
        text_file.write('Standard Deviation: {} \n'.format(np.std(min_losses)))
        
    return model

# Dataset Splitting
def split_proper_kf(train_ids, num_folds):
    folds_train_imgs = []
    folds_val_imgs = []
    folds_train_inds = []
    folds_val_inds = [] 
    img_names = []
    for i in train_ids:
        img_names.append(i[:6] + '/' + i.split('_')[1][2:])
    img_names = list(set(img_names))
    train_ids = np.array(train_ids)
    img_names = np.array(img_names)
    skf = KFold(n_splits = num_folds, random_state = 111, shuffle = True)
    print('Running {}-Fold data split'.format(num_folds))
    fold_number = 1
    for train_index, test_index in skf.split(img_names):
        print('Split dataset for fold:', fold_number)
        train_split, val_split = img_names[train_index], img_names[test_index]
        to_train = []
        to_val = []
        for i, img in enumerate(train_ids):
            orig = img[:6] + '/' + img.split('_')[1][2:]
            if orig in train_split:
                to_train.append(img)
            if orig in val_split:
                to_val.append(img)
        to_train = list(set(to_train))
        to_val = list(set(to_val))
        assert (len(list(set(to_train).intersection(set(to_val))))) == 0
        print('Number of training set images: {}, validation set images: {}'.format(len(to_train), len(to_val)))
        folds_train_imgs.append(to_train)
        folds_val_imgs.append(to_val)
        train_classes = pd.Series(to_train).apply(lambda x: x[:6]).value_counts()
        valid_classes = pd.Series(to_val).apply(lambda x: x[:6]).value_counts()
        print('Training classes counts:', train_classes)
        print('Validation classes counts:', valid_classes)
        print('Number of training set images: {}, validation set images: {}'.format(len(to_train), len(to_val)))
        inds_train = []
        inds_val = []
        for i, val in enumerate(train_ids):
            for j in to_train:
                if j in val:
                    inds_train.append(i)
        inds_val = list(set(range(len(train_ids))).difference(set(inds_train)))
        folds_train_inds.append(inds_train)
        folds_val_inds.append(inds_val)
        fold_number += 1
    return folds_train_imgs, folds_val_imgs, folds_train_inds, folds_val_inds

def split_proper_skf(train_ids, num_folds):
    folds_train_imgs = []
    folds_val_imgs = []
    folds_train_inds = []
    folds_val_inds = [] 
    img_names = []
    for i in train_ids:
        img_names.append(i[:6] + '/' + i.split('_')[1][2:])
    img_names = np.array(list(set(img_names)))
    train_ids = np.array(train_ids)
    classes = pd.Series(img_names).apply(lambda x: x[:6])
    skf = StratifiedKFold(n_splits = num_folds, random_state = 111, shuffle = True)
    print('Running {}-Fold data split'.format(num_folds))
    fold_number = 1
    for train_index, test_index in skf.split(img_names, classes):
        print('Split dataset for fold:', fold_number)
        train_split, val_split = img_names[train_index], img_names[test_index]
        to_train = []
        to_val = []
        for i, img in enumerate(train_ids):
            orig = img[:6] + '/' + img.split('_')[1][2:]
            if orig in train_split:
                to_train.append(img)
            if orig in val_split:
                to_val.append(img)
        to_train = list(set(to_train))
        to_val = list(set(to_val))
        assert (len(list(set(to_train).intersection(set(to_val))))) == 0
        print('Number of training set images: {}, validation set images: {}'.format(len(to_train), len(to_val)))
        folds_train_imgs.append(to_train)
        folds_val_imgs.append(to_val)
        train_classes = pd.Series(to_train).apply(lambda x: x[:6]).value_counts()
        valid_classes = pd.Series(to_val).apply(lambda x: x[:6]).value_counts()
        print('Training classes counts:', train_classes, '\n')
        print('Validation classes counts:', valid_classes, '\n')
        print('Number of training set images: {}, validation set images: {}'.format(len(to_train), len(to_val)))
        inds_train = []
        inds_val = []
        for i, val in enumerate(train_ids):
            for j in to_train:
                if j in val:
                    inds_train.append(i)
        inds_val = list(set(range(len(train_ids))).difference(set(inds_train)))
        folds_train_inds.append(inds_train)
        folds_val_inds.append(inds_val)
        fold_number += 1
    return folds_train_imgs, folds_val_imgs, folds_train_inds, folds_val_inds
