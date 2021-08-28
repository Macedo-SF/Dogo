import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential #importing our deep learing libraries
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.backend import argmax
import os
from collections import Counter

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

#model 2
model=tf.keras.Sequential([
tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),
tf.keras.layers.MaxPooling2D(2,2),

tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),

tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),

tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512,activation='softmax')

])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

checkpoint_path = "model_2_save/cp-"+str(28).zfill(4)+".ckpt"
model.load_weights(checkpoint_path)

loss, acc = model.evaluate(testing_data, verbose=2)
print("Epoch {:02d}:".format(28))
print("Accuracy: {:5.4f}".format(acc))
print("Loss: {:5.4f}".format(loss))

#having some fun
def testing_image(image_directory):
    test_image = image.load_img(image_directory, target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(x = test_image)
    prediction=result.argmax()
    if(prediction==1):
        return 'Dog'
    else:
        return 'Cat'

dir="./Shenanigans"
list_dir=os.listdir(dir)
for item in list_dir:
    print(item)
    print(testing_image(dir+'/'+item)) #prediction

dir="./dataset/test/dogs"
list_dir=os.listdir(dir)

dog=[]
for item in list_dir:
    #print(item)
    dog.append(testing_image(dir+'/'+item))

dir="./dataset/test/cats"
list_dir=os.listdir(dir)

cat=[]
for item in list_dir:
    #print(item)
    cat.append(testing_image(dir+'/'+item))

print(Counter(dog))
print(Counter(cat))
