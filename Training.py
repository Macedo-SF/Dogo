import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential #importing our deep learing libraries
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

#importing data
train_dir = 'dataset/train'
test_dir = 'dataset/test'
cats_train_dir = train_dir + '/cats'
cats_test_dir = test_dir + '/cats'
dogs_train_dir = train_dir + '/dogs'
dogs_test_dir = test_dir + '/dogs'

print('number of cats training images - ',len(os.listdir(cats_train_dir)))
print('number of dogs training images - ',len(os.listdir(dogs_train_dir)))
print('number of cats testing images - ',len(os.listdir(cats_test_dir)))
print('number of dogs testing images - ',len(os.listdir(dogs_test_dir)))

"modelo 1"

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

class_names = training_data.class_indices
print(class_names)

#saving the model
checkpoint_path = "model_1_save/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
#a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1,
                                                 save_freq=int(steps_per_epoch)*1)#save every epoch

#creating model
model = Sequential() #making our CNN
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = training_data.image_shape))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.3))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.2))
model.add(Conv2D(filters = 126, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.15))
model.add(Flatten())
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dropout(rate = 0.15))
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dropout(rate = 0.1))
model.add(Dense(units = len(set(training_data.classes)), activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.summary() #check model
#save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

fitted_model = model.fit_generator(training_data,steps_per_epoch = steps_per_epoch,epochs = epochs,
                                   validation_data = testing_data,validation_steps = validation_steps,
                                   callbacks=[cp_callback])

#testing model
def testing_image(image_directory):
    test_image = image.load_img(image_directory, target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(x = test_image)
    print(result)
    if result[0][0]  == 1:
        prediction = 'Dog'
    else:
        prediction = 'Cat'
    return prediction

print(testing_image(test_dir + '/cats/cat.4003.jpg')) #prediction
print(os.listdir(checkpoint_dir)) #saved weights

#save to fig
acc = fitted_model.history['accuracy']
val_acc = fitted_model.history['val_accuracy']

loss = fitted_model.history['loss']
val_loss = fitted_model.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig("Figures/Teste Modelo 1.png",dpi=400)

model.save('saved_models/model_1')

"modelo 2"

#saving the model
checkpoint_path = "model_2_save/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
#a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1,
                                                 save_freq=int(steps_per_epoch)*1)#save every epoch

#creating model
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
model.summary() #check model
#save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

fitted_model = model.fit_generator(training_data,steps_per_epoch = steps_per_epoch,epochs = epochs,
                                   validation_data = testing_data,validation_steps = validation_steps,
                                   callbacks=[cp_callback])

#testing model
def testing_image(image_directory):
    test_image = image.load_img(image_directory, target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(x = test_image)
    print(result)
    if result[0][0]  == 1:
        prediction = 'Dog'
    else:
        prediction = 'Cat'
    return prediction

print(testing_image(test_dir + '/cats/cat.4003.jpg')) #prediction
print(os.listdir(checkpoint_dir)) #saved weights

#save to fig
acc = fitted_model.history['accuracy']
val_acc = fitted_model.history['val_accuracy']

loss = fitted_model.history['loss']
val_loss = fitted_model.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig("Figures/Teste Modelo 2.png",dpi=400)

model.save('saved_models/model_2')