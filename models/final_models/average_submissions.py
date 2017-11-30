import os
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


def average_all_final():
    to_average_subnames = [x for x in os.listdir(sub_src) if 'FINAL_RUN' in x]
    sub_list = []
    for i in to_average_subnames:
        sub_list.append(pd.read_csv(sub_src + i))
    sub_name = 'FinalRun_6bestPooling'
    print(sub_list)
    avg = average_more_subs(sub_list)
    avg.to_csv(sub_src + '{}.csv'.format(sub_name), index = False)
    return

def average_best_final():
    to_average_subnames = os.listdir(sub_best_src)
    sub_list = []
    for i in to_average_subnames:
        sub_list.append(pd.read_csv(sub_best_src + i))
    sub_name = 'FINAL1_best5Pooling'
    avg = average_more_subs(sub_list)
    avg.to_csv(sub_src + '{}.csv'.format(sub_name), index = False)
    return

sub_src = '/media/w/1c392724-ecf3-4615-8f3c-79368ec36380/DS Projects/Kaggle/Intel_Cervix/submissions/' 
sub_best_src = sub_src + 'Best_pooling/'
average_all_final()
#average_best_final()
