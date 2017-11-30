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
from yolo_utils_processing import *

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
        st(iaa.Crop(percent=(0, 0.1))),
        st(iaa.GaussianBlur((0, 1.1))),
        st(iaa.Dropout((0.0, 0.03), per_channel=0.5)),
        st(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1), per_channel=0.5)),

        st(iaa.Add((-10, 15), per_channel=0.1)),
        st(iaa.Multiply((0.95, 1.2), per_channel=0.2)),
        st(iaa.ContrastNormalization((0.95, 1.1), per_channel=0.2)),
        st(iaa.Affine(
                #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                scale=(0.95, 1.15), 
                rotate=(-45, 45), 
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

src = '/media/w/1c392724-ecf3-4615-8f3c-79368ec36380/DS Projects/Kaggle/Intel_Cervix/data/training_data/train_crops_vgg11_299_oversampled/'

augment_train(src)
