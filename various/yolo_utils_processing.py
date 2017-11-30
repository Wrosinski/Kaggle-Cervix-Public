import cv2
import numpy as np
import pandas as pd
import os
import glob
import datetime
import time
import shutil
import matplotlib.pyplot as plt
from scipy import misc, ndimage 
import warnings
warnings.filterwarnings("ignore")

import imgaug as ia
from imgaug import augmenters as iaa


def savetest(files_path, save_path, savename):
    files = os.listdir(files_path)
    full_filenames = []
    for i in files:
        j = files_path + i
        full_filenames.append(j)
    print(full_filenames[0:5])
    with open(save_path + '{}.txt'.format(name), "a+") as myfile:
        for line in full_filenames:
            myfile.write(line + '\n')
    return

def savetrain(trainpath, save_path, savename):
    full_train = []
    for path, subdirs, files in os.walk(trainpath):
        for name in files:
            full_train.append(os.path.join(path, name))
    with open(save_path + '{}.txt'.format(savename), "a+") as myfile:
        for line in full_train:
            myfile.write(line + '\n')
    return


def get_im_cv2(path):
    #print(path)
    shape = cv2.imread(path).shape
    return shape

def load_test(path1):
    t = time.time()
    path = path1
    os.chdir(path)
    files = os.listdir(path)
    X_test = []
    X_test_id = []
    for fl in files:
        flbase = path + str(fl)
        try:
            img = get_im_cv2(fl)
            X_test.append(img)
            X_test_id.append(flbase)
        except Exception:
            print('Failed for image:', fl)
    print('Time it took to load test data: {}'.format(time.time() - t))
    X_test = pd.DataFrame(X_test)
    X_test_id = pd.DataFrame(X_test_id)
    testdf = pd.concat([X_test, X_test_id], axis = 1)
    testdf = testdf.iloc[:, [0, 1, -1]]
    testdf.columns = ['height', 'width', 'filename']
    print('Images loaded from: {}'.format(path1), '\n', '\n', testdf.head(), '\n', '\n')
    return testdf

def load_train(path1):
    X_train = []
    X_train_id = []
    y_train = []
    filenames = []
    failed = []
    t = time.time()
    folders = ['Type_1', 'Type_2', 'Type_3']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join(path1, fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            try:
                img = get_im_cv2(fl)
                X_train.append(img)
                X_train_id.append(flbase)
                y_train.append(fld)
                filenames.append(fl)
            except Exception:
                print('Failed for image:', flbase)
    print('Time it took to load train data: {}'.format(time.time() - t))
    X_train = pd.DataFrame(X_train)
    X_train_id = pd.DataFrame(X_train_id)
    testdf = pd.concat([X_train, X_train_id], axis = 1)
    traindf = testdf.iloc[:, [0, 1, -1]]
    traindf.columns = ['height', 'width', 'img_name']
    traindf['filename'] = filenames
    traindf['class'] = y_train
    print('Images loaded from: {}'.format(path1), '\n', '\n', testdf.head(), '\n', '\n')
    print(failed)
    return traindf


def load_boxes(results_name, imgs_df):
    res = pd.read_csv('{}'.format(results_name), sep = ',')
    bbs = res.iloc[:, [0, -4, -3, -2, -1]]
    bbs.rename(columns = {'image' : 'filename'}, inplace = True)
    full_bb = pd.merge(bbs, imgs_df, how = 'left', on = 'filename')
    x1 = full_bb['width'] * full_bb['xmin']
    x2 = full_bb['width'] * full_bb['xmax']
    y1 = full_bb['height'] * full_bb['ymin']
    y2 = full_bb['height'] * full_bb['ymax']
    coords = pd.concat([x1, x2, y1, y2], axis = 1)
    coords.columns = ['x1', 'x2', 'y1', 'y2']
    print('Bounding Boxes results loaded from: {}'.format(results_name), '\n', '\n', full_bb.head(), '\n', '\n')
    return full_bb, coords

def crop(bb_df, train = False):
    xs = bb_df['xmin'] * bb_df['width']
    xe = bb_df['xmax'] * bb_df['width']
    ys = bb_df['ymin'] * bb_df['height']
    ye = bb_df['ymax'] * bb_df['height']
    filename = bb_df['filename']
    if train:
        img_name = bb_df['img_name']
        classes = bb_df['class']
    if train:
        croppeddf = pd.concat([filename, img_name, classes, xs, xe, ys, ye], axis = 1)
        croppeddf.columns = [ 'filename', 'img_name', 'class', 'x1', 'x2', 'y1', 'y2']
    else:
        croppeddf = pd.concat([filename, xs, xe, ys, ye], axis = 1)
        croppeddf.columns = [ 'filename', 'x1', 'x2', 'y1', 'y2']
    return croppeddf

def make_dirs(path, labels):
    os.chdir(path)
    for i in labels:
        if i not in os.listdir(path):
            os.mkdir(i)
    return

def print_crops(cropped_images, img_range, images_number):
    rands = np.random.randint(0, img_range, size = images_number)
    for i in rands:
        plt.figure()
        im1 = ndimage.imread(cropped_images['filename'][i], mode = 'RGB')
        plt.imshow(im1)
        x1, x2, y1, y2 = int(cropped_images['x1'][i]), int(cropped_images['x2'][i]), int(cropped_images['y1'][i]), int(cropped_images['y2'][i])
        crop_img = im1[y1:y2, x1:x2]
        diff_w = x2 - x1
        diff_h = y2 - y1
        if diff_h > diff_w:
            crop_img = np.rot90(crop_img)
        plt.imshow(crop_img)
    return

def normalized(rgb):
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)
    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]
    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)
    return norm


def augment_train(tr_croppath):
    orig_path = tr_croppath
    tr_mine = load_train(tr_croppath)
    tr_classes = tr_mine['class'].value_counts()
    biggest_cls = tr_classes.keys()[0]
    num_imgs_biggest_cls = tr_classes.max()
    classes = ['Type_1', 'Type_2', 'Type_3']
    print('Class to remove: {}'.format(biggest_cls), '\n')
    classes.remove(biggest_cls)
    print('Classes to augment: {}'.format(classes))
    
    st = lambda aug: iaa.Sometimes(0.75, aug)
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        st(iaa.Crop(percent=(0, 0.2))),
        st(iaa.GaussianBlur((0, 1.5))),
        st(iaa.Dropout((0.0, 0.07), per_channel=0.5)),
        st(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.15), per_channel=0.5)),

        st(iaa.Add((-10, 15), per_channel=0.1)),
        st(iaa.Multiply((0.95, 1.2), per_channel=0.2)),
        st(iaa.ContrastNormalization((0.95, 1.15), per_channel=0.2)),
        st(iaa.Affine(
                #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                scale=(0.95, 1.15), 
                rotate=(-90, 90), 
            ))],
        random_order = True
        )

    for cls in classes:
        print('\n', 'Augmenting class: {}'.format(cls))
        tr_cls = tr_mine[tr_mine['class'] == cls]
        tr_cls.reset_index(inplace = True)
        batches_todo = int(num_imgs_biggest_cls/len(tr_cls))
        print('Batches to do: {}'.format(batches_todo), '\n')
        imgs_to_augment = []
        imgs_filenames = []
        for i in range(len(tr_cls)):
            img = cv2.imread('{}{}/{}'.format(orig_path, cls, tr_cls['img_name'][i]))
            imgs_to_augment.append(img)
            imgs_filenames.append(tr_cls['img_name'][i])
        for batch_idx in range(batches_todo):
            print('Augmenting for batch: {}'.format(batch_idx + 1))
            aug_imgs = seq.augment_images(imgs_to_augment)
            for i in range(len(aug_imgs)):
                cv2.imwrite(orig_path + '{0}/{1}_augbatch{2}_{3}.jpg'.format(cls, imgs_filenames[i][:-4], str(batch_idx + 1), str(i)), 
                            aug_imgs[i])
    return
