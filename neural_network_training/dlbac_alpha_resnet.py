# -*- coding: utf-8 -*-

from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
import numpy as np
from numpy import loadtxt
from tensorflow.keras.utils import to_categorical

from sklearn import metrics
import os
import sys

debug = True

param_count = len(sys.argv)

trainDataFileName = str(sys.argv[1])
testDataFileName = str(sys.argv[2])

batch_size = 16  # trained all networks with batch_size=16

# format of the dataset
# <uid rid> <8-13 user-metadata values> <8-13 resource-metadata values> <4 operations>
# load the train dataset
raw_train_dataset = loadtxt(trainDataFileName, delimiter=' ', dtype=np.str)
cols = raw_train_dataset.shape[1]
train_dataset = raw_train_dataset[:,2:cols] # TO SKIP UID RID

# load the test dataset
raw_test_dataset = loadtxt(testDataFileName, delimiter=' ', dtype=np.str)
test_dataset = raw_test_dataset[:,2:cols] # TO SKIP UID RID

# columns after removing uid/rid
cols = train_dataset.shape[1]
if debug:
  print('Total columns:', cols)

# determine number of metadata to be hide
# we will expose first eight user and first eight resource metadata to the model
# there are four operations
# 8 + 8 + 4 = 20

if cols > 20:
    hide_meta_data = cols - 20
else:
    hide_meta_data = 0
print('metadata to be hide: ', hide_meta_data)

# Compute depth and number of epochs based on metadata hide
# We use more deeper network for the dataset where metadata needs to hide
# If the dataset needs to hide metadata, then the depth of network is 56, otherwise 8
# The value of n helps to determine the depth of network
if hide_meta_data > 0:
    n = 9
else:
    n = 1

depth = n * 6 + 2

# we need less epoch for the deeper network
if depth > 8:
  epochs = 30
else:
  epochs = 60

# Model name, depth and version
model_type = 'ResNet%d' % (depth)
if debug:
  print('ResNet model type:', model_type)


# number of metadata
metadata = cols - 4

umeta_end = 8
rmeta_end = 16
umeta_hide_end = umeta_end + hide_meta_data
rmeta_hide_end = rmeta_end + hide_meta_data

# split x, y from train dataset
x_train = train_dataset[:,0:metadata]
y_train = train_dataset[:,metadata:cols].astype(int)

# hide (remove) user metadata after first eight metadata
x_train = np.delete(x_train, slice(umeta_end, umeta_hide_end), 1)
# hide (remove) resource metadata after first eight resource metadata
x_train = np.delete(x_train, slice(rmeta_end, rmeta_hide_end), 1)
if debug:
  print('User/resource metadata after meta data removal:', x_train.shape[1])

# split x, y from test dataset
x_test = test_dataset[:,0:metadata]
y_test = test_dataset[:,metadata:cols].astype(int)

# hide (remove) user/resource metadata after first eight of user/resource metadata
x_test = np.delete(x_test, slice(umeta_end, umeta_hide_end), 1)
x_test = np.delete(x_test, slice(rmeta_end, rmeta_hide_end), 1)
if debug:
  print('User/resource metadata after meta data removal:', x_test.shape[1])

############### OneHot ENCODING ##############
x_train = to_categorical(x_train)
x_test = to_categorical(x_test)

if debug:
  print('shape of x_train after encoding', x_train.shape)
  print('shape of x_test after encoding', x_test.shape)
#######################################

#determine batch size
batch_size = min(x_train.shape[0]/10, batch_size)
if debug:
  print('batch size: ' + str(batch_size))

# adding an extra dimension to make the input appropriate for ResNet
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

if debug:
  print('shape of x_train after adding new dimension', x_train.shape)
  print('shape of x_test after adding new dimension', x_test.shape)


def lr_schedule_resnet8(epoch):
    lr = 1e-3
    if epoch > 59:
        lr *= 1e-3
    elif epoch > 39:
        lr *= 1e-2
    elif epoch > 19:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def lr_schedule_resnet8_up(epoch):
    lr = 1e-3
    if epoch > 29:
        lr *= 1e-3
    elif epoch > 19:
        lr *= 1e-2
    elif epoch > 9:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=4):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU

    full = GlobalAveragePooling2D()(x)
    out = Dense(num_classes, activation='sigmoid', kernel_initializer='he_normal')(full)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=out)

    return model


input_shape = x_train.shape[1:]

dlbac_alpha = resnet_v1(input_shape=input_shape, depth=depth)

if depth > 8:
  dlbac_alpha.compile(loss='binary_crossentropy',
              optimizer=Adam(lr_schedule_resnet8_up(0)),
              metrics=['binary_accuracy'])
  lr_scheduler = LearningRateScheduler(lr_schedule_resnet8_up)
else:
  dlbac_alpha.compile(loss='binary_crossentropy',
              optimizer=Adam(lr_schedule_resnet8(0)),
              metrics=['binary_accuracy'])
  lr_scheduler = LearningRateScheduler(lr_schedule_resnet8)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [lr_reducer, lr_scheduler]

outputFileName = 'dlbac_alpha'
DIR_ASSETS = 'results/'
PATH_MODEL = DIR_ASSETS + outputFileName + '.hdf5'

history = dlbac_alpha.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=callbacks)

if debug:
  print('Saving trained dlbac_alpha to {}.'.format(PATH_MODEL))
if not os.path.isdir(DIR_ASSETS):
    os.mkdir(DIR_ASSETS)
dlbac_alpha.save(PATH_MODEL)

#save history to separate file
import pickle

PATH_HISTORY_FILE = DIR_ASSETS + 'history_' + outputFileName
with open(PATH_HISTORY_FILE, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

RESULT_FILE = DIR_ASSETS + 'result.txt'
result_file = open(RESULT_FILE, 'w+')
result_file.write('train data file name:%s\n' % (trainDataFileName))
result_file.write('test data file name:%s\n' % (testDataFileName))

scores = dlbac_alpha.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
'ResNet%d' % (depth)
result_file.write('Test loss:%f\n' % (scores[0]))
print('Test accuracy:', scores[1])
result_file.write('Test accuracy:%f\n' % (scores[1]))


# measure True Positive/ Negative, False Positive/ Negative
from sklearn import metrics
from sklearn.metrics import precision_score, confusion_matrix

y_preds = dlbac_alpha.predict(x_test)
y_preds = (y_preds > 0.5).astype(int)

g_tn = 0
g_fp = 0
g_fn = 0
g_tp = 0

# Measure True Positive/ Negative, False Positive/ Negative for each operation, 
# then combine it to measure actual counts
# we calculate the FPR, FNR offline

for i in range(4):
  tn, fp, fn, tp = confusion_matrix(y_test[:, i:i+1], y_preds[:, i:i+1]).ravel()
  print('tn: %s, fp: %s, fn: %s, tp: %s', tn, fp, fn, tp)
  g_tn = g_tn + tn
  g_fp = g_fp + fp
  g_fn = g_fn + fn
  g_tp = g_tp + tp
print('gtn: %s, gfp: %s, gfn: %s, gtp: %s', g_tn, g_fp, g_fn, g_tp)
result_file.write('TN: %s, FP: %s, FN: %s, TP: %s' % (g_tn, g_fp, g_fn, g_tp))
result_file.close()

