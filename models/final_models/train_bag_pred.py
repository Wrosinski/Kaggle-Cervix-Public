import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
        bagged_model_run(model_savename, params['num_bags'], model_name, num_epochs, batch_size = None, split = params['split_data'], prepare_sub = params['prepare_submission'], full_dst = params['train_src'], train_dst = 'train_set_gpu0', valid_dst = 'valid_set_gpu0', test_src = params['test_src'])
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


current_run = 'FINAL_RUN'
current_num_bags = 5
num_epochs = 50

bag2 = {
    'model_name': 'inception_globalavgpool',
    'train_src': 'train_crops_frcnn_299_combined',
    'test_src': 'test_crops_full_frcnn_299',
    'run_name': current_run,
    'num_bags': current_num_bags,
    'run_training': False,
    'prepare_submission': False,
    'only_prepare_submission': True,
    'split_data': True,
    }

bag3 = {
    'model_name': 'inception_globalavgpool',
    'train_src': 'train_crops_yolo_299_combined',
    'test_src': 'test_crops_full_yolo_299',
    'run_name': current_run,
    'num_bags': current_num_bags,
    'run_training': False,
    'prepare_submission': False,
    'only_prepare_submission': True,
    'split_data': True,
    }

bag4 = {
    'model_name': 'inception_globalavgpool',
    'train_src': 'train_crops_vgg_299_combined',
    'test_src': 'test_crops_full_vgg_299',
    'run_name': current_run,
    'num_bags': current_num_bags,
    'run_training': False,
    'prepare_submission': False,
    'only_prepare_submission': True,
    'split_data': True,
    }

bag5 = {
    'model_name': 'inception_avgpool',
    'train_src': 'train_crops_frcnn_299_combined',
    'test_src': 'test_crops_full_frcnn_299',
    'run_name': current_run,
    'num_bags': current_num_bags,
    'run_training': False,
    'prepare_submission': False,
    'only_prepare_submission': True,
    'split_data': True,
    }

bag6 = {
    'model_name': 'inception_avgpool',
    'train_src': 'train_crops_yolo_299_combined',
    'test_src': 'test_crops_full_yolo_299',
    'run_name': current_run,
    'num_bags': current_num_bags,
    'run_training': False,
    'prepare_submission': False,
    'only_prepare_submission': True,
    'split_data': True,
    }


if options.bagging == 1 and options.bagging_params == 1:
    parametrized_bagging(bag1)
if options.bagging == 1 and options.bagging_params == 2:
    parametrized_bagging(bag2)
if options.bagging == 1 and options.bagging_params == 3:
    parametrized_bagging(bag3)
if options.bagging == 1 and options.bagging_params == 4:
    parametrized_bagging(bag4)
if options.bagging == 1 and options.bagging_params == 5:
    parametrized_bagging(bag5)
if options.bagging == 1 and options.bagging_params == 6:
    parametrized_bagging(bag6)
if options.bagging == 1 and options.bagging_params == 7:
    parametrized_bagging(bag7)
if options.bagging == 1 and options.bagging_params == 8:
    parametrized_bagging(bag8)
if options.bagging == 1 and options.bagging_params == 9:
    parametrized_bagging(bag9)
if options.bagging == 1 and options.bagging_params == 10:
    parametrized_bagging(bag10)
if options.bagging == 1 and options.bagging_params == 11:
    parametrized_bagging(bag11)
