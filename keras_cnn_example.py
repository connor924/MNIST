#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:41:11 2017

@author: cmcglynn
"""
'''
MNIST CNN Keras Tutorial from https://elitedatascience.com/keras-tutorial-deep-learning-in-python
Using TensorFlow as backend
'''
import numpy as np
np.random.seed(924)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.datasets import mnist
from matplotlib import pyplot as plt


#load mnist data from keras datasets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

####pre-process input data

#1.reshape with depth of 1 (typically a full color image with 3 RGB channels would have depth 3. MNIST data has depth of 1)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
#2. convert data type to float32 and normalize data values to the range [ 0, 1]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


####pre-process class labels
# the y_train and y_test data are not split into the 10 distinct class labels, but instead a single array with the class values
# this requires some pre-processing so that the class labels are showing which of the 10 distinct class labels it is
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


###define model architecture
#declare sequential model
model= Sequential()
#declare input layer
model.add(Convolution2D(32, 3, 3, activation ='relu', input_shape=(28, 28, 1)))
#input shape here is(width,height,depth) of each image... 32 convolution filters, 3 rows in convolution kernel, 3 columns in conv. kernel

#add more layers
model.add(Convolution2D(32,3,3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#maxpooling2d is a way to reduce number of param. by sliding a 2x2 pooling filter across prev. layer and taking the max of the 4 values
#dropout layer is for regularizing model to prevent overfitting

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#For Dense layers, first parameter is output size of the layer.. Keras will automatically handle connects between the layers
#Flatten makes the weights from the convolutional layers 1-dimensional

#### Compile model
#define optimizer, loss function, and any metrics that should be displayed
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])


#### Fit model
model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)


#### Evaluate testing data
score = model.evaluate(X_test, Y_test, verbose=0)
aac