import cv2
import numpy as np
import pandas as pd
import os
import glob
import datetime
import time
import shutil
import matplotlib.pyplot as plt
import random
from PIL import Image
import warnings
#import imgaug as ia
#from imgaug import augmenters as iaa
from scipy import misc, ndimage
warnings.filterwarnings("ignore")
from processing_utils import *

size = (299, 299)

def normalized(rgb):
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)
    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]
    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)
    return norm

def make_crops_train(cropped_images, path, traindf, train_path = None, augment = False, normalize = False):
    pad_col = [0, 0, 0]
    saved_imgs = []
    cropped_imgs = []
    cropped_labels = []
    crop_filenames = cropped_images['img_name']
    labels = cropped_images['class']
    labels_set = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for i in range(len(cropped_images)):
        img = ndimage.imread(cropped_images['filename'][i], mode = 'RGB')
        x1, x2, y1, y2 = int(cropped_images['x1'][i]), int(cropped_images['x2'][i]), int(cropped_images['y1'][i]), int(cropped_images['y2'][i])
        crop_img = img[y1:y2, x1:x2]
        h, w = crop_img.shape[0], crop_img.shape[1]
        if h < 30 and w < 30:
            continue
        else:
            if h > w:
                crop_img = np.rot90(crop_img)
            if h > size[1] and w > size[0]:
                res_img = cv2.resize(crop_img, size, cv2.INTER_AREA)
                if normalize:
                    res_img = normalized(res_img)
                try:
                    final = Image.fromarray((res_img).astype(np.uint8))
                    final.save(path + '{0}/{1}_{2}'.format(labels[i], str(i), crop_filenames[i]))
                except KeyError:
                    print('Saving failed for image: ', crop_filenames[i])
                cropped_labels.append(labels[i])
                cropped_imgs.append(res_img)
            else:
                res_img = cv2.resize(crop_img, size, cv2.INTER_CUBIC)
                if crop_filenames[i] in saved_imgs:
                    continue
                else:
                    if normalize:
                        res_img = normalized(res_img)
                    try:
                        final = Image.fromarray((res_img).astype(np.uint8))
                        final.save(path + '{0}/{1}_{2}'.format(labels[i], str(i), crop_filenames[i]))
                    except KeyError:
                        print('Saving failed for image: ', crop_filenames[i])
                    cropped_labels.append(labels[i])
                    cropped_imgs.append(res_img)
    crop_set = set(crop_filenames)
    test_set = set(traindf['img_name'])
    testsetdiff = list(test_set.difference(crop_set))
    print('Number of train data set difference images:', len(testsetdiff))
    return 


def make_crops_test(cropped_images, path, test = False, test_path = None, normalize = False):
    pad_col = [0, 0, 0]
    saved_imgs = []
    testfiles = os.listdir(test_path)
    crop_filenames = cropped_images['filename'].str[-13:].tolist()
    crop_set = set(crop_filenames)
    
    for i in range(len(cropped_images)):
        img = cv2.imread(cropped_images['filename'][i])
        path = path
        if test:
            test_filename = cropped_images['filename'][i][-13:-4]
            copy_filename = cropped_images['filename'][i][-13:]
        x1, x2, y1, y2 = int(cropped_images['x1'][i]), int(cropped_images['x2'][i]), int(cropped_images['y1'][i]), int(cropped_images['y2'][i])
        crop_img = img[y1:y2, x1:x2]
        h, w = crop_img.shape[0], crop_img.shape[1]
        if h < 30 and w < 30:
            print('Crop {} omitted'.format(cropped_images['filename'][i]))
            continue
        else:
            if h > w:
                crop_img = np.rot90(crop_img)
            if h > size[1] and w > size[0]:
                res_img = cv2.resize(crop_img, size, cv2.INTER_AREA)
                if normalize:
                    res_img = normalized(res_img)
                if test:
                    cv2.imwrite(path + test_filename + '_' + str(i) + '.jpg'  , res_img)
                else:
                    cv2.imwrite(path + str(i) + '.jpg', res_img)
            else:
                res_img = cv2.resize(crop_img, size, cv2.INTER_CUBIC)
                if normalize:
                    res_img = normalized(res_img)
                if test:
                    if test_filename in saved_imgs:
                        continue
                    else:
                        cv2.imwrite(path + test_filename + '_' + str(i) + '.jpg'  , res_img)
                else:
                    cv2.imwrite(path + test_filename + '_' + str(i) + '.jpg' , res_img)
    
    def recover_crops(normalize = False):
        print('\n', 'Recover lost YOLO crops')
        recovered_path = '/home/w/DS_Projects/Kaggle/Nature Conservancy/Scripts/Detector/Mine/Craig_test_dataset/recovered_test_bbox/'
        original_filenames = os.listdir(recovered_path)
        filenames = pd.DataFrame(original_filenames)
        bag_nums = filenames[0].str[0:4]
        img_names1 = filenames[0].str[5:-4]
        img_names2 = filenames[0].str[5:]
        filenames = img_names1 + '_' + bag_nums
        set_recovered = set(img_names2)
        diff_recovered = list(set_recovered.difference(crop_set))
        print('Number of test data set recovered images:', len(diff_recovered))
        for new in diff_recovered:
            for orig in original_filenames:
                if new in orig:
                    diff_img = cv2.imread(recovered_path + orig)
                    diff_img_resized = cv2.resize(diff_img, size, cv2.INTER_LINEAR)
                    if normalize:
                        diff_img_resized = normalized(diff_img_resized)
                    cv2.imwrite(path + orig[5:-4] + '_recovered_' + orig[:4] + '.jpg', diff_img_resized)
                    
        crops_u = crop_set.union(set_recovered)
        return crops_u
        
    def recover_origtest(normalize = False):
        print('\n', 'Recover lost YOLO crops from original TensorBox test sets')
        recovered_path = '/home/w/DS_Projects/Kaggle/Nature Conservancy/Scripts/Detector/Mine/Craig_test_dataset/test_stg2_tensorbox_bags/'
        folders = [x for x in os.listdir(recovered_path) if 'bag' in x and '.json' not in x]
        img_names2 = []
        original_filenames = []
        for fld in folders:
            path_load = os.path.join(recovered_path, fld, '*')
            files = glob.glob(path_load)
            files = pd.Series(files)
            files_split = files.str.split('/')
            for i in range(len(files_split)):
                original_filenames.append('{}/'.format(fld) + files_split[i][-1])
                img_names2.append(files_split[i][-1][0:9] + '.jpg')
        set_recovered = set(img_names2)
        diff_recovered = list(set_recovered.difference(crop_set))
        origrecs = 0
        print('Number of test data set recovered images:', len(diff_recovered))
        for recovered_img in diff_recovered:
            new = recovered_img[:-4]
            for orig in original_filenames:
                if new in orig:
                    img_load_path = recovered_path + orig
                    diff_img = cv2.imread(img_load_path)
                    diff_img_resized = cv2.resize(diff_img, size, cv2.INTER_LINEAR)
                    if normalize:
                        diff_img_resized = normalized(diff_img_resized)
                    newimg_savename = new + '_recovered_fromCraigtest_' + str(origrecs) + '.jpg'
                    cv2.imwrite(path + newimg_savename, diff_img_resized)
                    origrecs += 1
        crops_u = crop_set.union(set_recovered)
        return crops_u
    
    
    def set_diff(normalize = False):
        print('\n', 'Take set difference between raw images and crops')
        test_set = set(testfiles)
        testsetdiff = list(test_set.difference(crops_u))
        print('Number of test data set difference images:', len(testsetdiff))
        for i in testsetdiff:
            diff_img = cv2.imread(test_path + i)
            diff_img_resized = cv2.resize(diff_img, size, cv2.INTER_LINEAR)
            if normalize:
                        diff_img_resized = normalized(diff_img_resized)
            cv2.imwrite(path + i[:-4] + '_origtest.jpg', diff_img_resized)
        return
    
    #crops_u = recover_crops()
    crops_u = recover_origtest(False)
    set_diff(False)
    return
            
        
def make_crops_test_pad(cropped_images, path, test = False, test_path = None):
    pad_col = [0, 0, 0]
    saved_imgs = []
    margins = 0
    testfiles = os.listdir(test_path)
    crop_filenames = cropped_images['filename'].str[-13:].tolist()
    omitted = 0
    for i in range(len(cropped_images)):
        img = cv2.imread(cropped_images['filename'][i])
        path = path
        if test:
            test_filename = cropped_images['filename'][i][-13:-4]
            copy_filename = cropped_images['filename'][i][-13:]
        x1, x2, y1, y2 = int(cropped_images['x1'][i]), int(cropped_images['x2'][i]), int(cropped_images['y1'][i]), int(cropped_images['y2'][i])
        crop_img = img[y1:y2, x1:x2]
        h, w, channels = crop_img.shape
        diff_w = x2 - x1
        diff_h = y2 - y1
        area = diff_w * diff_h

        if diff_h > diff_w:
            crop_img = np.rot90(crop_img)
        if h > size[1] and w > size[0]:
            res_img = cv2.resize(crop_img, size, cv2.INTER_AREA)
            cv2.imwrite(path + test_filename + '_' + str(i) + '.jpg'  , res_img)
        if h > size[1] and w < size[0]:
            diff_w = size[0] - w
            res_img = cv2.copyMakeBorder(crop_img,0,0,int(diff_w/2),int(diff_w/2),cv2.BORDER_CONSTANT,value=pad_col)
            res_img = cv2.resize(res_img, size, cv2.INTER_CUBIC)
            cv2.imwrite(path + test_filename + '_' + str(i) + '.jpg'  , res_img)
        if h < size[1] and w > size[0]:
            diff_h = size[1] - h
            res_img = cv2.copyMakeBorder(crop_img,int(diff_h/2),int(diff_h/2),0,0,cv2.BORDER_CONSTANT,value=pad_col)
            res_img = cv2.resize(res_img, size, cv2.INTER_CUBIC)
            cv2.imwrite(path + test_filename + '_' + str(i) + '.jpg'  , res_img)
        if h < size[1] and w < size[0]:
            diff_h = size[1] - h
            diff_w = size[0] - w
            res_img = cv2.copyMakeBorder(crop_img,int(diff_h/2),int(diff_h/2),int(diff_w/2),int(diff_w/2),cv2.BORDER_CONSTANT,value=pad_col)
            res_img = cv2.resize(res_img, size, cv2.INTER_CUBIC)
            cv2.imwrite(path + test_filename + '_' + str(i) + '.jpg'  , res_img)
        else:
            res_img = cv2.resize(crop_img, size, cv2.INTER_CUBIC)
            #print('Margin test case, image_shape: {}'.format(crop_img.shape))
            cv2.imwrite(path + test_filename + '_' + str(i) + '.jpg'  , res_img)
            margins += 1
    print('Number of margins cases: {}'.format(margins))
    print('Number of omitted images due to threshold: {}'.format(omitted))
    crop_set = set(crop_filenames)
    
    def recover_origtest():
        print('\n', 'Recover lost YOLO crops from original TensorBox test sets')
        recovered_path = '/home/w/DS_Projects/Kaggle/Nature Conservancy/Scripts/Detector/Mine/Craig test/Craig_cropping/'
        folders = [x for x in os.listdir(recovered_path) if 'bag' in x and '.json' not in x]
        img_names2 = []
        original_filenames = []
        for fld in folders:
            path_load = os.path.join(recovered_path, fld, '*')
            files = glob.glob(path_load)
            files = pd.Series(files)
            files_split = files.str.split('/')
            for i in range(len(files_split)):
                original_filenames.append('{}/'.format(fld) + files_split[i][-1])
                img_names2.append(files_split[i][-1][0:9] + '.jpg')
        set_recovered = set(img_names2)
        diff_recovered = list(set_recovered.difference(crop_set))
        origrecs = 0
        print('Number of test data set recovered images:', len(diff_recovered))
        for recovered_img in diff_recovered:
            new = recovered_img[:-4]
            for orig in original_filenames:
                if new in orig:
                    img_load_path = recovered_path + orig
                    diff_img = cv2.imread(img_load_path)
                    diff_img_resized = cv2.resize(diff_img, size, cv2.INTER_LINEAR)
                    newimg_savename = new + '_recovered_fromCraigtest_' + str(origrecs) + '.jpg'
                    cv2.imwrite(path + newimg_savename, diff_img_resized)
                    origrecs += 1
        crops_u = crop_set.union(set_recovered)
        return crops_u
    
    def recover_crops():
        print('\n', 'Recover lost YOLO crops')
        recovered_path = '/home/w/DS_Projects/Kaggle/Nature Conservancy/Scripts/Detector/Mine/Craig_test_dataset/recovered_test_bbox/'
        original_filenames = os.listdir(recovered_path)
        filenames = pd.DataFrame(original_filenames)
        bag_nums = filenames[0].str[0:4]
        img_names1 = filenames[0].str[5:-4]
        img_names2 = filenames[0].str[5:]
        filenames = img_names1 + '_' + bag_nums
        set_recovered = set(img_names2)
        diff_recovered = list(set_recovered.difference(crop_set))
        print('Number of test data set recovered images:', len(diff_recovered))
        for new in diff_recovered:
            for orig in original_filenames:
                if new in orig:
                    diff_img = cv2.imread(recovered_path + orig)
                    diff_img_resized = cv2.resize(diff_img, size, cv2.INTER_LINEAR)
                    cv2.imwrite(path + orig[5:-4] + '_recovered_' + orig[:4] + '.jpg', diff_img_resized)
        crops_u = crop_set.union(set_recovered)
        return crops_u
        
    def set_diff():
        print('\n', 'Take set difference between raw images and crops')
        test_set = set(testfiles)
        testsetdiff = list(test_set.difference(crops_u))
        print('Number of test data set difference images:', len(testsetdiff))
        for i in testsetdiff:
            diff_img = cv2.imread(test_path + i)
            diff_img_resized = cv2.resize(diff_img, size, cv2.INTER_LINEAR)
            cv2.imwrite(path + i[:-4] + '_origtest.jpg', diff_img_resized)
        return
    
    #crops_u = recover_origtest()
    crops_u = recover_crops()
    set_diff()
    return
            
def make_crops_train_pad(cropped_images, path, traindf, test_path = None, augment = False):
    pad_col = [0, 0, 0]
    saved_imgs = []
    cropped_imgs = []
    cropped_labels = []
    crop_filenames = cropped_images['img_name']
    labels = cropped_images['class']
    labels_set = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    omitted = 0
    for i, img_name in enumerate(cropped_images['img_name']):
        img = cv2.imread(cropped_images['filename'][i])
        x1, x2, y1, y2 = int(cropped_images['x1'][i]), int(cropped_images['x2'][i]), int(cropped_images['y1'][i]), int(cropped_images['y2'][i])
        crop_img = img[y1:y2, x1:x2]
        h, w, channels = crop_img.shape
        diff_w = x2 - x1
        diff_h = y2 - y1
        if diff_h > diff_w:
            crop_img = np.rot90(crop_img)
        margins = 0
        if h > size[1] and w > size[0]:
            res_img = cv2.resize(crop_img, size, cv2.INTER_AREA)
            cv2.imwrite(path + '{0}/{1}_{2}'.format(labels[i], str(i), crop_filenames[i]), res_img)
            cropped_labels.append(labels[i])
            cropped_imgs.append(res_img)
        if h > size[1] and w < size[0]:
            diff_w = size[0] - w
            res_img = cv2.copyMakeBorder(crop_img,0,0,int(diff_w/2),int(diff_w/2),
                                         cv2.BORDER_CONSTANT,value=pad_col)
            res_img = cv2.resize(res_img, size, cv2.INTER_CUBIC)
            cv2.imwrite(path + '{0}/{1}_{2}'.format(labels[i], str(i), crop_filenames[i]), res_img)
            cropped_labels.append(labels[i])
            cropped_imgs.append(res_img)
        if h < size[1] and w > size[0]:
            diff_h = size[1] - h
            res_img = cv2.copyMakeBorder(crop_img,int(diff_h/2),int(diff_h/2),0,0,
                                         cv2.BORDER_CONSTANT,value=pad_col)
            res_img = cv2.resize(res_img, size, cv2.INTER_CUBIC)
            cv2.imwrite(path + '{0}/{1}_{2}'.format(labels[i], str(i), crop_filenames[i]), res_img)
            cropped_labels.append(labels[i])
            cropped_imgs.append(res_img)
        if h < size[1] and w < size[0]:
            diff_h = size[1] - h
            diff_w = size[0] - w
            res_img = cv2.copyMakeBorder(crop_img,int(diff_h/2),int(diff_h/2),int(diff_w/2),int(diff_w/2),
                                     cv2.BORDER_CONSTANT,value=pad_col)
            res_img = cv2.resize(res_img, size, cv2.INTER_CUBIC)
            cv2.imwrite(path + '{0}/{1}_{2}'.format(labels[i], str(i), crop_filenames[i]), res_img)
            cropped_labels.append(labels[i])
            cropped_imgs.append(res_img)
        else:
            res_img = cv2.resize(crop_img, size, cv2.INTER_CUBIC)
            #print('Margin train case, image_shape: {}'.format(crop_img.shape))
            cv2.imwrite(path + '{0}/{1}_{2}'.format(labels[i], str(i), crop_filenames[i]), res_img)
            margins += 1
    print('Number of margins cases: {}'.format(margins))
    print('Number of omitted images due to threshold: {}'.format(omitted))
    crop_set = set(crop_filenames)
    test_set = set(traindf['img_name'])
    testsetdiff = list(test_set.difference(crop_set))
    print('Number of train data set difference images:', len(testsetdiff))
    return 

        
        
def augment_train(tr_croppath):
    orig_path = tr_croppath
    tr_mine = load_train(tr_croppath)
    tr_classes = tr_mine['class'].value_counts()
    biggest_cls = tr_classes.keys()[0]
    num_imgs_biggest_cls = tr_classes.max()
    fish_cls = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
    print('Class to remove: {}'.format(biggest_cls), '\n')
    fish_cls.remove(biggest_cls)
    print('Classes to augment: {}'.format(fish_cls))
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
        
    for cls in fish_cls:
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
                cv2.imwrite(orig_path + '{0}/aug_{1}_batch{2}_{3}'.format(cls, str(i), str(batch_idx + 1), imgs_filenames[i]), 
                            aug_imgs[i])
    return



def random_nof(tr_origpath, tr_croppath, tiles, normalize = False):
    INPATH = tr_origpath + 'NoF'
    OUTPATH = tr_croppath + 'NoF'
    dx = size[0]
    dy = size[1]
    tilesPerImage = tiles
    files = os.listdir(INPATH)
    numOfImages = len(files)
    t = time.time()
    for file in files:
        with Image.open(os.path.join(INPATH, file)) as im:
            for i in range(1, tilesPerImage+1):
                newname = file[:9]+'_box'+str(i)+'.jpg'
                w, h = im.size
                x = random.randint(0, w-dx-1)
                y = random.randint(0, h-dy-1)
                if normalize:
                    im = np.array(im)
                    im = normalized(im)
                    im = Image.fromarray(np.uint8(im))
                im.crop((x,y, x+dx, y+dy))\
                 .save(os.path.join(OUTPATH, newname))
    t = time.time()-t
    print("Done {} images in {:.2f}s".format(numOfImages, t))
    print("({:.1f} images per second)".format(numOfImages/t))
    print("({:.1f} tiles per second)".format(tilesPerImage*numOfImages/t)) 
    return
