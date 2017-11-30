from keras.layers.core import Flatten, Dense, Dropout, Reshape, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adadelta, Adam, SGD, Nadam, RMSprop
from keras.layers import merge
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception 

img_size = (299, 299)
size = img_size + (3,)
learning_rate = 0.0001
optimizer = Adam(lr = learning_rate, decay = 1e-3)


# ResNet 50
def resnet_dense():
    resnet_notop = ResNet50(include_top=False, weights='imagenet',
                        input_tensor=None, input_shape = size)
    output = resnet_notop.get_layer(index = -1).output
    output = Flatten(name='flatten')(output)
    output = Dropout(0.2)(output)
    output = Dense(512)(output)
    output = PReLU()(output)
    output = Dropout(0.3)(output)
    output = Dense(3, activation='softmax', name = 'predictions')(output)
    Resnet_model = Model(resnet_notop.input, output)
    Resnet_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return Resnet_model

# Xception
def xception_globalavgpool():
    Xception_notop = Xception(include_top=False, weights='imagenet',
                        input_tensor=None, input_shape = size)
    output = Xception_notop.get_layer(index = -1).output
    output = GlobalAveragePooling2D()(output)
    output = Dense(3, activation='softmax', name='predictions')(output)
    Xception_model = Model(Xception_notop.input, output)
    Xception_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return Xception_model

def xception_avgpool():
    Xception_notop = Xception(include_top=False, weights='imagenet',
                        input_tensor=None, input_shape = size)
    output = Xception_notop.get_layer(index = -1).output
    output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
    output = Flatten(name='flatten')(output)
    output = Dense(3, activation='softmax', name = 'predictions')(output)
    Xception_model = Model(Xception_notop.input, output)
    Xception_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return Xception_model

def xception_globalavgpool_last14():
    Xception_notop = Xception(include_top=False, weights='imagenet',
                        input_tensor=None, input_shape = size)
    output = Xception_notop.get_layer(index = -1).output
    output = GlobalAveragePooling2D()(output)
    output = Dense(3, activation='softmax', name='predictions')(output)
    Xception_model = Model(Xception_notop.input, output)
    for layer in Xception_model.layers[:-14]:
        layer.trainable = False
    for layer in Xception_model.layers[-14:]:
        layer.trainable = True
    Xception_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return Xception_model


# Inception V3
def inception_globalavgpool():
    InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
                        input_tensor=None, input_shape = size)
    output = InceptionV3_notop.get_layer(index = -1).output
    output = GlobalAveragePooling2D()(output)
    output = Dense(3, activation='softmax', name='predictions')(output)
    InceptionV3_model = Model(InceptionV3_notop.input, output)
    InceptionV3_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return InceptionV3_model

def inception_avgpool():
    InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
                        input_tensor=None, input_shape = size)
    output = InceptionV3_notop.get_layer(index = -1).output
    output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
    output = Flatten(name='flatten')(output)
    output = Dense(3, activation='softmax', name = 'predictions')(output)
    InceptionV3_model = Model(InceptionV3_notop.input, output)
    InceptionV3_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return InceptionV3_model

def inception_avgpool_dense():
    InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
                        input_tensor=None, input_shape = size)
    output = InceptionV3_notop.get_layer(index = -1).output
    output = AveragePooling2D((4, 4), name='avg_pool')(output)
    output = Flatten(name='flatten')(output)
    output = Dropout(0.2)(output)
    output = Dense(512)(output)
    output = PReLU()(output)
    output = Dropout(0.3)(output)
    output = Dense(3, activation='softmax', name = 'predictions')(output)
    InceptionV3_model = Model(InceptionV3_notop.input, output)
    InceptionV3_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return InceptionV3_model

def inception_dense():
    InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
                        input_tensor=None, input_shape = size)
    output = InceptionV3_notop.get_layer(index = -1).output
    output = Flatten(name='flatten')(output)
    output = Dropout(0.2)(output)
    output = Dense(512)(output)
    output = PReLU()(output)
    output = Dropout(0.3)(output)
    output = Dense(3, activation='softmax', name = 'predictions')(output)
    InceptionV3_model = Model(InceptionV3_notop.input, output)
    InceptionV3_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return InceptionV3_model
