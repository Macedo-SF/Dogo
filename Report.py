import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential #importing our deep learing libraries
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys

#importing data
train_dir = 'dataset/train'
test_dir = 'dataset/test'
cats_train_dir = train_dir + '/cats'
cats_test_dir = test_dir + '/cats'
dogs_train_dir = train_dir + '/dogs'
dogs_test_dir = test_dir + '/dogs'

#preparing data
data_generator = ImageDataGenerator(rescale = 1. / 250,zoom_range = 0.2) #RGB photo to array
#parameters
batch_size = 100
epochs=30
steps_per_epoch=np.ceil(8000/batch_size)
validation_steps=np.ceil(2000/batch_size)

training_data = data_generator.flow_from_directory(directory = train_dir,target_size = (150,150),
                                                   batch_size = batch_size,class_mode  = 'binary')
testing_data = data_generator.flow_from_directory(directory = test_dir,target_size = (150,150),
                                                  batch_size = batch_size,class_mode  = 'binary')

#model 1
model_1 = Sequential() #making our CNN
model_1.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = training_data.image_shape))
model_1.add(MaxPooling2D(pool_size = (2, 2)))
model_1.add(Dropout(rate = 0.3))
model_1.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
model_1.add(MaxPooling2D(pool_size = (2, 2)))
model_1.add(Dropout(rate = 0.2))
model_1.add(Conv2D(filters = 126, kernel_size = (3, 3), activation = 'relu'))
model_1.add(MaxPooling2D(pool_size = (2, 2)))
model_1.add(Dropout(rate = 0.15))
model_1.add(Flatten())
model_1.add(Dense(units = 32, activation = 'relu'))
model_1.add(Dropout(rate = 0.15))
model_1.add(Dense(units = 64, activation = 'relu'))
model_1.add(Dropout(rate = 0.1))
model_1.add(Dense(units = len(set(training_data.classes)), activation = 'softmax'))
model_1.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#open report file
sys.stdout = open('./Reports/model_1.txt','w')
for counter in range(31):
    sys.stdout.close()
    checkpoint_path = "model_1_save/cp-"+str(counter).zfill(4)+".ckpt"
    model_1.load_weights(checkpoint_path)

    loss, acc = model_1.evaluate(testing_data, verbose=2)
    sys.stdout = open('./Reports/model_1.txt','a')
    print("Epoch {:02d}:".format(counter))
    print("Accuracy: {:5.4f}".format(acc))
    print("Loss: {:5.4f}".format(loss))
sys.stdout.close()

#model 2
model_2=tf.keras.Sequential([
tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),
tf.keras.layers.MaxPooling2D(2,2),

tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),

tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),

tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512,activation='softmax')

])
model_2.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

sys.stdout = open('./Reports/model_2.txt','w')
for counter in range(31):
    sys.stdout.close()
    checkpoint_path = "model_2_save/cp-"+str(counter).zfill(4)+".ckpt"
    model_2.load_weights(checkpoint_path)

    loss, acc = model_2.evaluate(testing_data, verbose=2)
    sys.stdout = open('./Reports/model_2.txt','a')
    print("Epoch {:02d}:".format(counter))
    print("Accuracy: {:5.4f}".format(acc))
    print("Loss: {:5.4f}".format(loss))
sys.stdout.close()