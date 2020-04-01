# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 20:31:25 2020

@author: uni tech
"""


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping



(x_train, y_train) , (x_test, y_test) = keras.datasets.mnist.load_data()


#reshaping the images
x_train= x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test= x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing inputs and outputs
x_train /= 255
x_test /= 255


# Defining model and adding layers to the neural network.
model= Sequential()
model.add(Conv2D(32, input_shape=(28, 28, 1), kernel_size=(4,4), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(10, activation='softmax'))


# Defining a callback function
early_stopping= EarlyStopping(monitor='val_loss',
                              patience=5,
                              verbose=1,
                              restore_best_weights= True)

# Compiling the model
model.compile(optimizer= 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Training the model
model.fit(x_train, y_train, epochs=3,validation_split = 0.2 ,callbacks=[early_stopping])








