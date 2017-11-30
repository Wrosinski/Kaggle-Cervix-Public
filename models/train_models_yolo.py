import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import time
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
import selected_models
from optparse import OptionParser
from keras import backend as K
from keras.utils.np_utils import to_categorical
from models_utils_loading import *
from models_utils_train import *
from models_utils_skf import *
np.random.seed(1337)
tf.set_random_seed(1337)


parser = OptionParser()
parser.add_option("--bagging", dest="bagging", help="Run bagged model.", type = int, default = 0)
parser.add_option("--bagging_params", dest="bagging_params", help="Specify parameters for bagging.", type = int, default = 0)
parser.add_option("--skf", dest="skf", help="Run SKF model.", type = int, default = 0)
parser.add_option("--skf_params", dest="skf_params", help="Specify parameters for SKF run.", type = int, default = 0)
(options, args) = parser.parse_args()
    

def parametrized_bagging(params):
    print('\n', 'Running bagging with parameters:', params, '\n')
    model_name = getattr(selected_models, params['model_name'])
    model_savename = '{}_bagged_{}_{}'.format(params['model_name'],  params['train_src'], params['run_name'])

    if params['run_training'] == True:
        bagged_model_run(model_savename, params['num_bags'], model_name, 200, batch_size = None, split = params['split_data'], prepare_sub = params['prepare_submission'], full_dst = params['train_src'], train_dst = 'train_set_gpu1', valid_dst = 'valid_set_gpu1', test_src = params['test_src'])
    if params['only_prepare_submission'] == True:
        prep_submission_bag(model_savename, params['num_bags'], params['test_src'])
    return

def parametrized_skf(params):
    print('\n', 'Running SKF with parameters:', params, '\n')
    model_name = getattr(selected_models, params['model_name'])
    model_savename = '{}_{}foldSKF_{}_{}'.format(params['model_name'], params['n_folds'], params['train_src'], params['run_name'])
    
    src = '/media/w/1c392724-ecf3-4615-8f3c-79368ec36380/DS Projects/Kaggle/Intel_Cervix/data/training_data/'
    X, y, train_ids = load_train(src + '{}/'.format(params['train_src']))
    y = to_categorical(y)
    X_test, test_ids = load_test(src + '{}/{}/'.format(params['test_src'], params['test_src']))

    if params['run_training'] == True:
        skf_model_run(X, y, X_test, train_ids, params['n_folds'], params['stratify'], modelname = model_name, savename = model_savename)
    if params['prepare_submission'] == True:
        prep_submission_kf(model_savename, params['n_folds'], test_src = params['test_src'])
    return


params_bagging1 = {
    'model_name': 'xception_globalavgpool',
    'train_src': 'train_crops_yolo_299_oversampled',
    'test_src': 'test_crops_yolo_299',
    'run_name': 'FINAL1',
    'num_bags': 5,
    'run_training': True,
    'prepare_submission': True,
    'only_prepare_submission': False,
    'split_data': True,
    }

params_bagging2 = {
    'model_name': 'inception_globalavgpool',
    'train_src': 'train_crops_yolo_299_oversampled',
    'test_src': 'test_crops_yolo_299',
    'run_name': 'FINAL1',
    'num_bags': 5,
    'run_training': False,
    'prepare_submission': False,
    'only_prepare_submission': True,
    'split_data': True,
    }

params_bagging3 = {
    'model_name': 'resnet_globavgpool',
    'train_src': 'train_crops_yolo_299_oversampled',
    'test_src': 'test_crops_yolo_299',
    'run_name': 'FINAL1',
    'num_bags': 5,
    'run_training': True,
    'prepare_submission': True,
    'only_prepare_submission': False,
    'split_data': True,
    }


if options.bagging == 1 and options.bagging_params == 1:
    parametrized_bagging(params_bagging1)
if options.bagging == 1 and options.bagging_params == 2:
    parametrized_bagging(params_bagging2)
if options.bagging == 1 and options.bagging_params == 3:
    parametrized_bagging(params_bagging3)
if options.bagging == 1 and options.bagging_params == 4:
    parametrized_bagging(params_bagging4)


params_skf1 = {
    'model_name': 'xception_globalavgpool',
    'train_src': 'train_crops_yolo_299_oversampled',
    'test_src': 'test_crops_yolo_299',
    'run_name': 'run1',
    'n_folds': 10,
    'run_training': False,
    'prepare_submission': True,
    'stratify': True,
    }

params_skf2 = {
    'model_name': 'resnet_dense',
    'train_src': 'train_crops_yolo_299_oversampled',
    'test_src': 'test_crops_yolo_299',
    'run_name': 'run1',
    'n_folds': 10,
    'run_training': True,
    'prepare_submission': True,
    'stratify': True,
    }

if options.skf == 1 and options.skf_params == 1:
    parametrized_skf(params_skf1)
if options.skf == 1 and options.skf_params == 2:
    parametrized_skf(params_skf2)

