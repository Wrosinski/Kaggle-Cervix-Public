import os
import glob
import cv2
import time
import scipy
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
from scipy import misc
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
np.random.seed(1337)


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
            y_train.append(index)
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train_id, y_train

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

def lgb_foldrun_test(X, y, X_test, params, name, save = True):
    skf = StratifiedKFold(n_splits = 5, random_state = 111, shuffle = True)
    if isinstance(X, pd.core.frame.DataFrame):
        X = X.values
    if isinstance(y, pd.core.frame.DataFrame):
        y = y.is_duplicate.values
    if isinstance(y, pd.core.frame.Series):
        y = y.values
    print('Running LGBM model with parameters:', params)
        
    i = 0
    losses = []
    oof_train = np.zeros((1481, 3))
    oof_test = np.zeros((512, 3, 5))
    for tr_index, val_index in skf.split(X, y):
        X_tr, X_val = X[tr_index], X[val_index]
        y_tr, y_val = y[tr_index], y[val_index]
        t = time.time()
        
        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_val = lgb.Dataset(X_val, y_val)
        print('Start training on fold: {}'.format(i))
        gbm = lgb.train(params, lgb_train, num_boost_round = 100000, valid_sets = lgb_val,
                        early_stopping_rounds = 200, verbose_eval = 100)
        print('Start predicting...')
        val_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
        oof_train[val_index, :] = val_pred
        score = log_loss(y_val, val_pred)
        losses.append(score)
        if X_test is not None:
            test_preds = gbm.predict(X_test, num_iteration=gbm.best_iteration)
            oof_test[:, :, i] = test_preds
        print('Final score for fold {} :'.format(i), score, '\n',
              'Time it took to train and predict on fold:', time.time() - t, '\n')
        i += 1
    print('Mean logloss for model in 5-folds SKF:', np.array(losses).mean(axis = 0))
    oof_train = pd.DataFrame(oof_train)
    oof_train.columns = ['Type_1', 'Type_2', 'Type_3']
    oof_test = oof_test.mean(axis = 2)
    oof_test = pd.DataFrame(oof_test)
    oof_test.columns = ['Type_1', 'Type_2', 'Type_3']
    if save:
        oof_train.to_pickle('OOF_preds/stacking_train/train_preds_{}.pkl'.format(name))
        oof_test.to_pickle('OOF_preds/stacking_test/test_preds_{}.pkl'.format(name))
    return oof_train, oof_test


def xgb_foldrun_test(X, y, X_test, params, name, save = True):
    skf = StratifiedKFold(n_splits = 5, random_state = 111, shuffle = True)
    if isinstance(X, pd.core.frame.DataFrame):
        X = X.values
    if isinstance(y, pd.core.frame.DataFrame):
        y = y.is_duplicate.values
    if isinstance(y, pd.core.frame.Series):
        y = y.values
    print('Running XGB model with parameters:', params)
    
    i = 0
    losses = []
    oof_train = np.zeros((1481, 3))
    oof_test = np.zeros((512, 3, 5))
    for tr_index, val_index in skf.split(X, y):
        X_tr, X_val = X[tr_index], X[val_index]
        y_tr, y_val = y[tr_index], y[val_index]
        t = time.time()
        
        dtrain = xgb.DMatrix(X_tr, label = y_tr)
        dval = xgb.DMatrix(X_val, label = y_val)
        watchlist = [(dtrain, 'train'), (dval, 'valid')]
        print('Start training on fold: {}'.format(i))
        gbm = xgb.train(params, dtrain, 100000, watchlist, 
                        early_stopping_rounds = 200, verbose_eval = 100)
        print('Start predicting...')
        val_pred = gbm.predict(xgb.DMatrix(X_val), ntree_limit=gbm.best_ntree_limit)
        oof_train[val_index, :] = val_pred
        score = log_loss(y_val, val_pred)
        losses.append(score)
        if X_test is not None:
            test_preds = gbm.predict(X_test, ntree_limit=gbm.best_ntree_limit)
            oof_test[:, :, i] = test_preds
        print('Final score for fold {} :'.format(i), score, '\n',
              'Time it took to train and predict on fold:', time.time() - t, '\n')
        i += 1
    print('Mean logloss for model in 5-folds SKF:', np.array(losses).mean(axis = 0))
    oof_train = pd.DataFrame(oof_train)
    oof_train.columns = ['Type_1', 'Type_2', 'Type_3']
    oof_test = oof_test.mean(axis = 2)
    oof_test = pd.DataFrame(oof_test)
    oof_test.columns = ['Type_1', 'Type_2', 'Type_3']
    if save:
        oof_train.to_pickle('OOF_preds/stacking_train/train_preds_{}.pkl'.format(name))
        oof_test.to_pickle('OOF_preds/stacking_test/test_preds_{}.pkl'.format(name))
    return oof_train, oof_test
