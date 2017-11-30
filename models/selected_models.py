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

#optimizer = SGD(lr=learning_rate, momentum = 0.9, decay = 1e-4, nesterov = True)
optimizer = Adam(lr = learning_rate, decay = 1e-3)


# ResNet50
def resnet_dense():
    resnet_notop = ResNet50(include_top=False, weights='imagenet',
                        input_tensor=None, input_shape = size)
    output = resnet_notop.get_layer(index = -1).output
    output = Flatten(name='flatten')(output)
    output = Dropout(0.2)(output)
    output = Dense(512)(output)
    output = PReLU()(output)
    output = BatchNormalization()(output)
    output = Dropout(0.3)(output)
    output = Dense(3, activation='softmax', name = 'predictions')(output)
    Resnet_model = Model(resnet_notop.input, output)
    Resnet_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return Resnet_model

def resnet_avgpool():
    resnet_notop = ResNet50(include_top=False, weights='imagenet',
                        input_tensor=None, input_shape = size)
    output = resnet_notop.get_layer(index = -1).output
    output = AveragePooling2D((7, 7), strides=(7, 7), name='avg_pool')(output)
    output = Flatten(name='flatten')(output)
    output = Dense(3, activation='softmax', name = 'predictions')(output)
    Resnet_model = Model(resnet_notop.input, output)
    Resnet_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return Resnet_model

#Best single results on bagged splits, around 0.55-0.59
def resnet_globavgpool():
    resnet_notop = ResNet50(include_top=False, weights='imagenet',
                        input_tensor=None, input_shape = size)
    output = resnet_notop.get_layer(index = -1).output
    output = GlobalAveragePooling2D()(output)
    output = Dense(3, activation='softmax', name = 'predictions')(output)
    Resnet_model = Model(resnet_notop.input, output)
    Resnet_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return Resnet_model

def resnet_max():
    resnet_notop = ResNet50(include_top=False, weights='imagenet',
                        input_tensor=None, input_shape = size)
    output = resnet_notop.get_layer(index = -2).output
    output = MaxPooling2D((7, 7), strides=(7, 7), name='max_pool')(output)
    output = Flatten(name='flatten')(output)
    output = Dropout(0.2)(output)
    output = Dense(512)(output)
    output = PReLU()(output)
    output = Dropout(0.3)(output)
    output = Dense(3, activation='softmax', name = 'predictions')(output)
    Resnet_model = Model(resnet_notop.input, output)
    Resnet_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return Resnet_model

def resnet_mix():
    resnet_notop = ResNet50(include_top=False, weights='imagenet',
                        input_tensor=None, input_shape = size)
    output = resnet_notop.get_layer(index = -2).output
    glob_pool = AveragePooling2D((7, 7), strides=(7, 7), name='avg_pool')(output)
    max_pool = MaxPooling2D((7, 7), strides=(7, 7), name='max_pool')(output)
    concat_pool = merge([glob_pool, max_pool], mode='concat', concat_axis=-1)
    output = Flatten(name='flatten')(concat_pool)
    output = Dropout(0.2)(output)
    output = Dense(512)(output)
    output = PReLU()(output)
    output = Dropout(0.3)(output)
    output = Dense(3, activation='softmax', name = 'predictions')(output)
    Resnet_model = Model(resnet_notop.input, output)
    Resnet_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return Resnet_model

# Xception
def xception_dense():
    notop = Xception(include_top=False, weights='imagenet',
                        input_tensor=None, input_shape = size)
    output = notop.get_layer(index = -1).output
    output = Flatten(name='flatten')(output)
    output = Dropout(0.2)(output)
    output = Dense(512)(output)
    output = PReLU()(output)
    output = BatchNormalization()(output)
    output = Dropout(0.3)(output)
    output = Dense(3, activation='softmax', name = 'predictions')(output)
    Xception_model = Model(notop.input, output)
    Xception_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return Xception_model

def xception_globalavgpool():
    Xception_notop = Xception(include_top=False, weights='imagenet',
                        input_tensor=None, input_shape = size)
    output = Xception_notop.get_layer(index = -1).output
    output = GlobalAveragePooling2D()(output)
    output = Dense(3, activation='softmax', name='predictions')(output)
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
def inception_dense():
    InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
                        input_tensor=None, input_shape = size)
    output = InceptionV3_notop.get_layer(index = -1).output
    output = Flatten(name='flatten')(output)
    output = Dropout(0.2)(output)
    output = Dense(512)(output)
    output = PReLU()(output)
    output = BatchNormalization()(output)
    output = Dropout(0.3)(output)
    output = Dense(3, activation='softmax', name = 'predictions')(output)
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

def inception_globalavgpool():
    InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
                        input_tensor=None, input_shape = size)
    output = InceptionV3_notop.get_layer(index = -1).output
    output = GlobalAveragePooling2D()(output)
    output = Dense(3, activation='softmax', name='predictions')(output)
    InceptionV3_model = Model(InceptionV3_notop.input, output)
    InceptionV3_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return InceptionV3_model

def inception_globalavgpool_172():
    InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
                        input_tensor=None, input_shape = size)
    output = InceptionV3_notop.get_layer(index = -1).output
    output = GlobalAveragePooling2D()(output)
    output = Dense(3, activation='softmax', name='predictions')(output)
    InceptionV3_model = Model(InceptionV3_notop.input, output)
    for layer in InceptionV3_model.layers[:172]:
        layer.trainable = False
    for layer in InceptionV3_model.layers[172:]:
        layer.trainable = True
    InceptionV3_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return InceptionV3_model

def inception_dense_172():
    InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
                        input_tensor=None, input_shape = size)
    output = InceptionV3_notop.get_layer(index = -1).output
    output = Flatten(name='flatten')(output)
    output = Dropout(0.2)(output)
    output = Dense(512)(output)
    output = PReLU()(output)
    output = BatchNormalization()(output)
    output = Dropout(0.3)(output)
    output = Dense(3, activation='softmax', name = 'predictions')(output)
    InceptionV3_model = Model(InceptionV3_notop.input, output)
    for layer in InceptionV3_model.layers[:172]:
        layer.trainable = False
    for layer in InceptionV3_model.layers[172:]:
        layer.trainable = True
    InceptionV3_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return InceptionV3_model

