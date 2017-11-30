import os

def savetest(files_path, save_path, savename):
    files = os.listdir(files_path)
    files = [x for x in files if '.jpg' in x]
    full_filenames = []
    for i in files:
        j = files_path + i
        full_filenames.append(j)
    print(full_filenames[0:5])
    with open(save_path + '{}.txt'.format(savename), "a+") as myfile:
        for line in full_filenames:
            myfile.write(line + '\n')
    return

def savetrain(trainpath, save_path, savename):
    full_train = []
    for path, subdirs, files in os.walk(trainpath):
        for name in files:
            full_train.append(os.path.join(path, name))
    full_train = [x for x in full_train if '.jpg' in x and 'to_delete' not in x and 'test' not in x]
    with open(save_path + '{}.txt'.format(savename), "a+") as myfile:
        for line in full_train:
            myfile.write(line + '\n')
    return 

trpath = '/media/w/1c392724-ecf3-4615-8f3c-79368ec36380/DS Projects/Kaggle/Intel_Cervix/data/train_additional/'
savepath = '/media/w/1c392724-ecf3-4615-8f3c-79368ec36380/DS Projects/Kaggle/Intel_Cervix/scripts/detector/YOLO/'
savetrain(trpath, savepath, 'cervix_full_additional_Paulfiltered')


tepath = '/media/w/1c392724-ecf3-4615-8f3c-79368ec36380/DS Projects/Kaggle/Intel_Cervix/data/test/'
#savetest(tepath, savepath, 'cervix_teststg1')
