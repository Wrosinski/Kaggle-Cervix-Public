import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

def average_two_subs(sub1, sub2):
    averaged = sub1.copy()
    averaged.iloc[:, :3] = (sub1.iloc[:, :3] + sub2.iloc[:, :3]) / 2
    return averaged

def average_more_subs(subs):
    averaged = subs[0]
    averaged.iloc[:, :3] = 0
    for i in range(len(subs)):
        averaged.iloc[:, :3] += subs[i].iloc[:, :3]
    averaged.iloc[:, :3] = normalize(averaged.iloc[:, :3], axis = 1, norm = 'l1')
    #averaged.iloc[:, :3] = averaged.iloc[:, :3].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis = 0)
    return averaged

sub_src = '/media/w/1c392724-ecf3-4615-8f3c-79368ec36380/DS Projects/Kaggle/Intel_Cervix/submissions/' 

name1 = 'xception_globalavgpool_5foldSKF_train_crops_frcnn_299_oversampled_avg1_grouped_25augs'
name2 = 'xception_globalavgpool_5foldSKF_train_crops_vgg_299_oversampled_avg1_grouped_25augs'
name3 = 'inception_globalavgpool_5foldSKF_train_crops_vgg_299_oversampled_avg1_grouped_25augs'
name4 = 'inception_globalavgpool_5foldSKF_train_crops_frcnn_299_oversampled_avg1_grouped_25augs'
name5 = 'resnet_dense_5foldSKF_train_crops_frcnn_299_oversampled_avg1_grouped_25augs'
name6 = 'resnet_dense_5foldSKF_train_crops_vgg_299_oversampled_avg1_grouped_25augs'

s1 = pd.read_csv(sub_src + name1 + '.csv')
s2 = pd.read_csv(sub_src + name2 + '.csv')
s3 = pd.read_csv(sub_src + name3 + '.csv')
s4 = pd.read_csv(sub_src + name4 + '.csv')
s5 = pd.read_csv(sub_src + name5 + '.csv')
s6 = pd.read_csv(sub_src + name6 + '.csv')
sub_list = [s1, s2, s3, s4, s5, s6]
sub_name = 'xceptionGlobalavgpool_&_inceptionGlobalavgpool_&resnetDense_5foldSKF_onFRCNN_&_VGG_25augs_norm'

#avg_two = average_two_subs(s1, s2)
#avg_two.to_csv(sub_src + name1 + '_' + name2 + '.csv', index = False)
avg = average_more_subs(sub_list)
avg.to_csv(sub_src + '{}.csv'.format(sub_name), index = False)
