import cv2
import os
import glob
import datetime
import time
import shutil
import imgaug as ia
import numpy as np
import pandas as pd
from scipy import misc, ndimage
from PIL import Image
from imgaug import augmenters as iaa

from various_utils_general import *
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


def fill_cropped_set_train(src_orig, src_crop, size):

    def get_orig_imgnames(src1):
        orig_imgnames = []
        folders = ['Type_1', 'Type_2', 'Type_3']
        for fld in folders:
            index = folders.index(fld)
            print('Load folder {} (Index: {})'.format(fld, index))
            dst = os.path.join(src1, fld, '*.jpg')
            files = glob.glob(dst)
            for fl in files:
                flbase = fld + '/' + os.path.basename(fl).split('.')[0]
                orig_imgnames.append(flbase)
        return orig_imgnames

    def get_cropped_imgnames(src2, test = False):
        orig_imgnames = []
        folders = ['Type_1', 'Type_2', 'Type_3']
        for fld in folders:
            index = folders.index(fld)
            print('Load folder {} (Index: {})'.format(fld, index))
            dst = os.path.join(src2, fld, '*.jpg')
            files = glob.glob(dst)
            for fl in files:
                if test:
                    flbase = fld + '/' + os.path.basename(fl).split('_')[0]
                else:
                    flbase = fld + '/' + os.path.basename(fl).split('_')[0][2:]
                orig_imgnames.append(flbase)
        return orig_imgnames

    def set_diff(src, dst, normalize = False):
        print('Number of test data set difference images:', len(testsetdiff))
        for i in testsetdiff:
            diff_img = ndimage.imread(src + i + '.jpg', mode = 'RGB')
            diff_img_resized = cv2.resize(diff_img, size, cv2.INTER_AREA)
            print(i)
            if normalize:
                diff_img_resized = normalized(diff_img_resized)
            final = Image.fromarray((diff_img_resized).astype(np.uint8))
            final.save(dst + '{0}/id{1}_original_{2}'.format(i.split('/')[0], i.split('/')[1], '.jpg'))
        return

    crop_set = set(get_cropped_imgnames(src_crop))
    test_set = set(get_orig_imgnames(src_orig))
    testsetdiff = list(test_set.difference(crop_set))
    set_diff(src_orig, src_crop)
    return


def fill_cropped_set_test(src_orig, src_crop, size):

    def get_orig_imgnames_test(src1):
        orig_imgnames = []
        files = glob.glob(src1 + '*.jpg')
        for fl in files:
            flbase = os.path.basename(fl).split('.')[0]
            orig_imgnames.append(flbase)
        return orig_imgnames

    def get_cropped_imgnames_test(src2):
        orig_imgnames = []
        files = glob.glob(src2 + '*.jpg')
        for fl in files:
            flbase = os.path.basename(fl).split('_')[0]
            orig_imgnames.append(flbase)
        return orig_imgnames

    def set_diff_test(src, dst, normalize = False):
        print('Number of test data set difference images:', len(testsetdiff))
        for i in testsetdiff:
            diff_img = ndimage.imread(src + i + '.jpg', mode = 'RGB')
            diff_img_resized = cv2.resize(diff_img, size, cv2.INTER_AREA)
            print(i)
            if normalize:
                diff_img_resized = normalized(diff_img_resized)
            final = Image.fromarray((diff_img_resized).astype(np.uint8))
            final.save(dst + '{0}_original{1}'.format(i, '.jpg'))
        return

    crop_set = set(get_cropped_imgnames_test(src_crop))
    test_set = set(get_orig_imgnames_test(src_orig))
    testsetdiff = list(test_set.difference(crop_set))
    set_diff_test(src_orig, src_crop)
    return

src_orig = '/media/w/1c392724-ecf3-4615-8f3c-79368ec36380/DS Projects/Kaggle/Intel_Cervix/data/train/'
src_orig_test = '/media/w/1c392724-ecf3-4615-8f3c-79368ec36380/DS Projects/Kaggle/Intel_Cervix/data/test/'

src_raw_train = '/media/w/1c392724-ecf3-4615-8f3c-79368ec36380/DS Projects/Kaggle/Intel_Cervix/data/additional_crops/'
src_crop = '/media/w/1c392724-ecf3-4615-8f3c-79368ec36380/DS Projects/Kaggle/Intel_Cervix/data/training_data/train_additional_crops_Florian_299_oversampled/'

src_crop_test = '/media/w/1c392724-ecf3-4615-8f3c-79368ec36380/DS Projects/Kaggle/Intel_Cervix/data/training_data/test_crops_frcnn_299/test_crops_frcnn_299/'

size = (299, 299)

#fill_cropped_set_test(src_orig_test, src_crop_test, size)
#fill_cropped_set_train(src_orig_test, src_crop_test, size)
#augment_train(src_raw_train)

#resize_train(src_raw_train, src_crop)
augment_train(src_crop)
