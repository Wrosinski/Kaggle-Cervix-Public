## Intel & MobileODT Cervical Cancer Screening Competition Code

### Detectors:

* [YOLO (Darknet)](https://pjreddie.com/darknet/yolo/) 
* [Faster R-CNN](https://github.com/yhenon/keras-frcnn) (FRCNN)
* Florian's Bounding Box Regression (not included)

    
Uncleaned code.

In terms of my detectors (Wojtek) - training Faster R-CNN is much less troubling.
So far models trained on Paul's annotations from forums seem to give the better results, crops made by Faster R-CNN are performing best in terms of local validation and LB in case of most models.


### Classifiers:

 * __train_models_gpu0/1__ - lets you specify most training parameters - bagging/SKF, keras model, crops for training and test sets, run name etc.
     * 'run_training' is for only training the model
     * 'prepare_submission' to prepare submission after training is done on specified number of bags/folds
     * 'only_prepare_submission' just run predictions with specified parameters
     * 'split_data' for bagging, if set to _True_, a new random split is created for every bag
     * for SKF run, source path for data folder must be specified in 'parametrized_skf' function - src = ''
     * I've got a set of folders created in data folder: 'train_set_gpu0', 'train_set_gpu1', 'valid_set_gpu0', 'valid_set_gpu1', in which data for each split is saved and then loaded from when bagging models. In SKF run data is loaded and split in-memory.

 * __models_utils_train__ - set of functions for bagged models and a few more for overall use such as submission generation etc.
     * in this file a few parameters are specified such as image size, test time augmentations number for either SKF ('test_aug_kf') or bagging ('test_aug_bag'), validation set size ('validation_size') and if for each bag a split with set 'random_state' parameter should be made ('set_seed').
     * paths should be specified:
         * 'data_src' - folder in which data is stored
         * 'sub_src' - folder for saving & loading raw submissions predicted by classifier on crops
         * 'sub_dst' - folder for saving final submission grouped by crop ID's
         * 'checks_src' - folder for saving model checkpoints and history

 * __models_utils_train__ - set of functions for SKF/KF splits
     * 'checks_src' - folder for saving model checkpoints and history
     * 'oof_src' - folder for saving OOF predictions for potential stacking if we'd like to try it

 * __selected_models__ - set of selected Keras classifiers performing quite well
     * so far 'xception_globalavgpool' and 'resnet_globalavgpool' seem to be performing best - best scores achieved on FRCNN crops, less than 0.6 logloss on 0.15 random split. those also performed best on LB (0.65-0.7 scores)
     * '_dense' models are unstable, on some splits they perform well, around 0.6-0.7 logloss, on others they fail to converge well being stuck at >1.0 loss
