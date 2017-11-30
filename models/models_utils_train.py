import os
import pickle
import h5py
import os
import glob
import shutil
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import backend as K
from keras.callbacks import EarlyStopping, History, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.models  import load_model
from keras.models import Model
from keras.applications.imagenet_utils import *
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from models_utils_loading import load_ids
np.random.seed(1337)
tf.set_random_seed(1337)


test_aug_bag = 25
test_aug_kf = 25
validation_size = 0.1
set_seed = False

img_size = (299, 299)
size = img_size + (3,)
img_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.1,
                zoom_range=0.25,
                rotation_range=45,
                width_shift_range=0.25,
                height_shift_range=0.25,
                horizontal_flip=True,
                channel_shift_range=0.07)
img_datagen_valid = ImageDataGenerator(rescale=1./255,)

data_src = '/media/w/1c392724-ecf3-4615-8f3c-79368ec36380/DS Projects/Kaggle/Intel_Cervix/data/training_data/'
sub_src = '/media/w/1c392724-ecf3-4615-8f3c-79368ec36380/DS Projects/Kaggle/Intel_Cervix/submissions/Raw/'
sub_dst = '/media/w/1c392724-ecf3-4615-8f3c-79368ec36380/DS Projects/Kaggle/Intel_Cervix/submissions/'
checks_src = '/media/w/1c392724-ecf3-4615-8f3c-79368ec36380/DS Projects/Kaggle/Intel_Cervix/scripts/models/checks/'


# Model Training
def bagged_model_run(savename, num_bags, train_model, epochs, batch_size = None, split = True, prepare_sub = False, check_name = None, full_dst = None, train_dst = None, valid_dst = None, test_src = None):
    t = time.time()
    print('\n', 'Working on model:', savename, '\n', 'Bagging for: {} bags'.format(num_bags), '\n')
    if 'inception' in savename:
        batch_size = 24
    if 'resnet' in savename:
        batch_size = 16
    if 'xception' in savename:
        batch_size = 12
        
    if check_name is not None:
        print('Loading model: ', check_name)
        bagged_train(name, num_bags, train_model, split = split, epochs_num = epochs, batch_size = batch_size,
            check_name = check_name, full_dst = full_dst, train_dst = train_dst, valid_dst = valid_dst)
        
    bagged_train(savename, num_bags, train_model, split = split, epochs_num = epochs, batch_size = batch_size, 
            full_dst = full_dst, train_dst = train_dst, valid_dst = valid_dst)
    
    print('Training took: ', time.time() - t)
    if prepare_sub:
        prep_submission_bag(savename, num_bags, test_src)
    return

def bagged_train(savename, num_bags, train_model, epochs_num, batch_size, check_name = None, split = False, full_dst = None, train_dst = None, valid_dst = None):
    
    min_losses = []
    loss_history = []
    os.makedirs('{}{}'.format(checks_src, savename), exist_ok = True)
    if 'dense' in savename:
        stop_patience = 8
    else:
        stop_patience = 5
            
    for bag in range(num_bags):
        print ('Training on bag: {}'.format(bag + 1))
        if split:
            print('Preparing new data split for bag:', bag + 1)
            full_train_path = data_src + full_dst
            train_path = data_src + train_dst
            val_path = data_src + valid_dst
            train_ids = load_ids(full_train_path)
            if set_seed:
                tr, val = split_by_id(train_ids, validation_size, seed = bag)
            else:
                tr, val = split_by_id(train_ids, validation_size)
            assert len(list((set(tr).intersection(set(val))))) == 0
            os.chdir(data_src)
            shutil.rmtree(train_dst)
            shutil.rmtree(valid_dst)
            os.makedirs(train_dst)
            os.makedirs(valid_dst)
            os.chdir('{}/{}'.format(data_src, train_dst))
            for cls in ['Type_1', 'Type_2', 'Type_3']: os.mkdir('../{}/'.format(train_dst) + cls)
            for cls in ['Type_1', 'Type_2', 'Type_3']: os.mkdir('../{}/'.format(valid_dst) + cls)
            save_id_split(tr, val, full_train_path, train_path, val_path)
            os.chdir('{}/{}'.format(data_src, train_dst))
        else:
            os.chdir('{}/{}'.format(data_src, train_dst))
        
        nb_train_samples = len(glob.glob('*/*.*'))
        nb_validation_samples = len(glob.glob('../{}/*/*.*'.format(valid_dst)))
        epochs = epochs_num
        random_seed = np.random.randint(0, 100000)
        classes = ['Type_1', 'Type_2', 'Type_3']
            
        train_datagen = img_datagen
        train_generator = train_datagen.flow_from_directory(
                '../{}/'.format(train_dst),
                target_size=img_size,
                batch_size=batch_size,
                seed = random_seed,
                shuffle = True,
                classes=classes,
                class_mode='categorical')

        valid_datagen = img_datagen_valid
        validation_generator = valid_datagen.flow_from_directory(
                '../{}/'.format(valid_dst),
                target_size=img_size,
                batch_size = batch_size,
                shuffle = True,
                classes=classes,
                class_mode='categorical')
        callbacks = [ModelCheckpoint('{}{}/{}_bag{}.h5'.format(checks_src, savename, savename, bag + 1), 
                                        monitor='val_loss', 
                                        verbose = 0, save_best_only = True),
                     EarlyStopping(monitor='val_loss', patience = stop_patience, verbose = 1),
                     CSVLogger('{}{}/{}_bag{}_history.csv'.format(checks_src, savename, savename, bag + 1), append = True),
                     ReduceLROnPlateau(monitor='val_loss', factor = 0.5, verbose = 1, 
                                  patience = 3, min_lr = 1e-5)
                     ]
                     
        if check_name is not None:
            model = load_model('{}/{}_bag{}.h5'.format(checks_src, check_name, bag + 1))
        else:
            model = train_model()
            
        history = model.fit_generator(
                train_generator,
                steps_per_epoch = nb_train_samples/batch_size,
                epochs=epochs,
                validation_data=validation_generator,
                validation_steps = nb_validation_samples/batch_size,
                callbacks=callbacks)
        
        validation_loss = history.history['val_loss']
        loss_history.append(validation_loss)
        min_losses.append(np.min(validation_loss))
        print('Minimum loss for current bag:', np.min(validation_loss), '\n',
              'Mean loss for current bag:', np.mean(validation_loss))
        
    print('Statistics for minimum losses per bag in current bagging run: \n',
          'Mean: {}'.format(np.mean(min_losses)), '\n',
          'Minimum: {}'.format(np.min(min_losses)), '\n',
          'Maximum: {}'.format(np.max(min_losses)), '\n',
          'Standard Deviation: {}'.format(np.std(min_losses)), '\n')
    with open('{}{}/{}_stats.txt'.format(checks_src, savename, savename), 'w') as text_file:
        text_file.write('Statistics for minimum losses per bag in current bagging run: \n')
        text_file.write('Minimum: {} \n'.format(np.min(min_losses)))
        text_file.write('Maximum: {} \n'.format(np.max(min_losses)))
        text_file.write('Mean: {} \n'.format(np.mean(min_losses)))
        text_file.write('Standard Deviation: {} \n'.format(np.std(min_losses)))
                      
    return


# Predicting Data
def predict_crops(name, num_bags, src = None):
    bs = 32
    bag_preds = []
    os.chdir(data_src)
    for bag in range(num_bags):
        try:
            print('\n','Predicting crops for bag: {}'.format(bag + 1))
            model = load_model('{}{}/{}_bag{}.h5'.format(checks_src, name, name, bag + 1))
            print('Model loaded.', '\n')
            nb_test_samples = len(glob.glob('{}/*/*.*'.format(src)))
            test_aug = test_aug_bag
            test_datagen = img_datagen
            for aug in range(test_aug):
                print('Predictions for Augmentation -', aug + 1)
                random_seed = np.random.randint(0, 100000)
                test_generator = test_datagen.flow_from_directory(
                        '{}/'.format(src),
                        target_size= img_size,
                        batch_size = bs,
                        shuffle = False,
                        seed = random_seed,
                        classes = None,
                        class_mode = None)
                test_image_list = test_generator.filenames
                if aug == 0:
                    predictions = model.predict_generator(test_generator, nb_test_samples/bs)
                else:
                    predictions += model.predict_generator(test_generator, nb_test_samples/bs)
            predictions /= test_aug
            #preds = do_clip(predictions, 0.98)
            bag_preds.append(predictions)
        except OSError:
            break
    print('Predictions on crops done.')
    return bag_preds, test_generator

def predict_crops_kf(name, num_folds, src = None):
    bs = 32
    bag_preds = []
    os.chdir(data_src)
    for fold in range(num_folds):
        try:
            print('\n','Predicting crops for fold: {}'.format(fold + 1))
            model = load_model('{}{}/{}_fold{}.h5'.format(checks_src, name, name, fold + 1))
            print('Model loaded.', '\n')
            nb_test_samples = len(glob.glob('{}/*/*.*'.format(src)))
            test_aug = test_aug_kf
            test_datagen = img_datagen
            for aug in range(test_aug):
                print('Predictions for Augmentation -', aug + 1)
                random_seed = np.random.randint(0, 100000)
                test_generator = test_datagen.flow_from_directory(
                        '{}/'.format(src),
                        target_size= img_size,
                        batch_size = bs,
                        shuffle = False,
                        seed = random_seed,
                        classes = None,
                        class_mode = None)
                test_image_list = test_generator.filenames
                if aug == 0:
                    predictions = model.predict_generator(test_generator, nb_test_samples/bs)
                else:
                    predictions += model.predict_generator(test_generator, nb_test_samples/bs)
            predictions /= test_aug
            preds = do_clip(predictions, 0.99)
            bag_preds.append(predictions)
        except OSError:
            break
    print('Predictions on crops done.')
    return bag_preds, test_generator


# Dataset Splitting
def split_by_id(train_ids, test_size, seed = None):
    print('Validation set size:', test_size)
    img_names = []
    classes = []
    for i in train_ids:
        img_names.append(i[:6] + '/' + i.split('_')[1][2:])
        classes.append(i[:6])
    img_names = list(set(img_names))
    classes = pd.Series(img_names).apply(lambda x: x[:6])
    if seed is not None:
        train_split, val_split = train_test_split(img_names, test_size = test_size, stratify = classes, random_state = seed)
    else:
        train_split, val_split = train_test_split(img_names, test_size = test_size, stratify = classes)
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
    return to_train, to_val

def save_id_split(trfile, valfile, full_path, train_path, valid_path):
    flds = ['Type_1', 'Type_2', 'Type_3']
    for i in flds:
        if i not in os.listdir(train_path):
            try:
                os.mkdir(i)
            except Exception as e:
                print(e)
        if i not in os.listdir(valid_path):
            try:
                os.mkdir(i)
            except Exception as e:
                print(e)
    for line in trfile:
        cols = line.split('/')
        src = "{}/{}/{}".format(full_path, cols[0], cols[1])
        dst = "{}/{}/{}".format(train_path, cols[0], cols[1])
        shutil.copy(src, dst)
    for line in valfile:
        cols = line.split('/')
        src = "{}/{}/{}".format(full_path, cols[0], cols[1])
        dst = "{}/{}/{}".format(valid_path, cols[0], cols[1])
        shutil.copy(src, dst)
    return


# Submission Utilities
def prep_submission_bag(name, num_bags, test_src):
    bag_preds_crops, test_generator_crop = predict_crops(name, num_bags, test_src)
    submission(bag_preds_crops, test_generator_crop, name)
    sub_full = read_sub('sample_submission')
    sub_crops = read_sub(name)
    sub_crops_mean = group_crops_sub(sub_crops)
    final_sub = average_subs(sub_full, sub_crops_mean, save = True, crop_name = '{}_grouped_{}augs'.format(name, test_aug_bag))
    return 

def prep_submission_kf(name, num_folds, test_src):
    bag_preds_crops, test_generator_crop = predict_crops_kf(name, num_folds, test_src)
    submission(bag_preds_crops, test_generator_crop, name)
    sub_full = read_sub('sample_submission')
    sub_crops = read_sub(name)
    sub_crops_mean = group_crops_sub(sub_crops)
    final_sub = average_subs(sub_full, sub_crops_mean, save = True, crop_name = '{}_grouped_{}augs'.format(name, test_aug_kf))
    return 

def submission(bag_preds, test_generator, name):
    bag_preds2 = np.array(bag_preds).mean(axis = 0)
    print('Begin to write submission file ..')
    f_submit = open(os.path.join(sub_src, '{}'.format(name) +'.csv'), 'w')
    f_submit.write('image,Type_1,Type_2,Type_3\n')
    for i, image_name in enumerate(test_generator.filenames):
        pred = ['%.6f' % p for p in bag_preds2[i, :]]
        if i%100 == 0:
            print(i, '/', 600)
        f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))
    f_submit.close()
    print('Submission {} written.'.format(name))
    return 

def submission_inmem(bag_preds, test_ids, name):
    bag_preds2 = np.array(bag_preds).mean(axis = 0)
    print('Begin to write submission file ..')
    f_submit = open(os.path.join(sub_src, '{}'.format(name) +'.csv'), 'w')
    f_submit.write('image,Type_1,Type_2,Type_3\n')
    for i, image_name in enumerate(test_ids):
        pred = ['%.6f' % p for p in bag_preds2[i, :]]
        if i%100 == 0:
            print(i, '/', 600)
        f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))
    f_submit.close()
    print('Submission {} written.'.format(name))
    return


def read_sub(name):
    sub = pd.read_csv(sub_src + '{}.csv'.format(name))
    return sub

def group_crops_sub(df):
    df['image_name'] = df['image'].apply(lambda x: x.split('_')[0] + '.jpg')
    df_mean = df.groupby(['image_name']).mean()
    df_mean.reset_index(inplace = True)
    return df_mean

def average_subs(df1, df2, crop_name, name = None, save = False, average = False):
    av_sub = pd.merge(df1, df2, on = 'image_name', how = 'left')
    classes = ['Type_1', 'Type_2', 'Type_3']
    for i in classes:
        av_sub['{}'.format(i)] = (av_sub['{}_x'.format(i)] + av_sub['{}_y'.format(i)]) / 2
    crops_sub = av_sub.iloc[:, -6:-3]
    crops_sub.columns = classes
    crops_sub['image_name'] = av_sub['image_name']
    print('Saving crops only predictions.')
    print(crops_sub.columns)
    if average:
        averaged_sub = av_sub.iloc[:, -6:]
        averaged_sub.columns = classes
        averaged_sub['image_name'] = av_sub['image_name']
        print('Saving averaged predictions.')
        print(averaged_sub.columns)
    if save:
        if average:
            averaged_sub.to_csv(sub_dst + '{}.csv'.format(crop_name), index = False)
        else:
            crops_sub.to_csv(sub_dst + '{}.csv'.format(crop_name), index = False)
    return 


def do_clip(arr, mx): 
    arr_clip = np.clip(arr, (1-mx)/7, mx)
    arr_clip[arr_clip >= mx] = 1
    return arr_clip

def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)
    return
          
 
