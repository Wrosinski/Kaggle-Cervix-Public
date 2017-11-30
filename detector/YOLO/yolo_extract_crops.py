import os
import cv2
import numpy as np
from cropping_utils import *
from processing_utils import *


def get_dfs(train = False, test = False):
    if test:
        testdf = load_test(te_origpath)
        bb_te, co_te = load_boxes(te_respath, testdf)
        cropte = crop(bb_te)
        cropte['img_name'] = cropte['filename'].str[-13:]
        return cropte
    if train:
        traindf = load_train(tr_origpath)
        bb_tr, co_tr = load_boxes(tr_respath, traindf)
        croptr = crop(bb_tr, True)
        return traindf, croptr
    
def dataset_crop(croptr = None, cropte = None, traindf = None, test = False, train = False, random_nof_crop = False, 
                 augment = False, pad = False, normalize = False):
    if train:
        if pad:
            make_crops_train_pad(croptr, tr_croppath, traindf, tr_origpath, augment = augment)
        else:
            make_crops_train(croptr, tr_croppath, traindf, tr_origpath, augment = True, normalize = normalize)
       
        if random_nof_crop:
            random_nof(tr_origpath, tr_croppath, tiles, normalize = True)
        else:
            nof_imgs = os.listdir(tr_origpath + 'NoF')
            crop_set = set(croptr)
            nof_test_set = set(nof_imgs)
            nof_testsetdiff = list(nof_test_set.difference(crop_set))
            for i in nof_testsetdiff:
                diff_img = cv2.imread(tr_origpath + 'NoF/' + i)
                diff_img_resized = cv2.resize(diff_img, size, cv2.INTER_LINEAR)
                if normalize:
                    diff_img_resized = normalized(diff_img_resized)
                cv2.imwrite(tr_croppath + 'NoF/' + i, diff_img_resized)
        if augment:
            augment_train(tr_croppath)
            
    if test:
        make_crops_test(cropte, te_croppath, True, te_origpath, normalize = normalize)
    return


def run_test(normalize = False):
    cropte = get_dfs(test = True)
    dataset_crop(cropte = cropte, test = True, normalize = normalize)
    return

def run_train(random_nof = True, augment = True, normalize = False):
    make_dirs(tr_croppath, labels_set)
    traindf, croptr = get_dfs(train = True)
    dataset_crop(croptr = croptr, traindf = traindf, train = True, random_nof_crop = random_nof, augment = augment, normalize = normalize)
    return


tr_origpath = '/home/w/DS_Projects/Kaggle/Nature Conservancy/train_full/'
tr_croppath = '/home/w/DS_Projects/Kaggle/Nature Conservancy/Scripts/Detector/Mine/Data/train_crops_aug_nb/'
tr_respath = '/home/w/Development/darknet2/darknet/train_full_betadded_res145kcraig.txt'

te_origpath = '/home/w/DS_Projects/Kaggle/Nature Conservancy/test_stg1/test_stg1/'
te_croppath = '/home/w/DS_Projects/Kaggle/Nature Conservancy/Scripts/Detector/Mine/Data/test_stg1_145k/test_stg1_145k/'
te_respath = '/home/w/Development/darknet2/darknet/test_orig_res145kcraig.txt'

labels_set = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
size = (299, 299)
tiles = 8

run_test()
#run_train(augment = True)
#random_nof(tr_origpath, tr_croppath, tiles)
