import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

# Training:
training = ImageDataGenerator(
        rescale = 1./255,
        shear_range= 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)
training_set = training.flow_from_directory(
        'training_flowers',
         target_size= (64,64),
        batch_size = 32,
        class_mode= 'categorical')

#Test:
test= ImageDataGenerator(rescale= 1./255)
test_set= (test.flow_from_directory
(
        'test_flowers',
        target_size= (64,64),
        batch_size = 32,
        class_mode = 'categorical')
)

#Model Building:
cnn = tf.keras.models.Sequential()

#Building Convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 4, activation = 'relu', input_shape=[64,64, 3] ))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 4, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides= 2))

cnn.add(tf.keras.layers.Dropout(0.5))
cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))

cnn.add(tf.keras.layers.Dense(units = 5, activation = 'softmax'))

cnn.compile(optimizer = 'rmsprop', loss ='categorical_crossentropy', metrics =['accuracy'])

Model = cnn.fit(x= training_set, validation_data = test_set, epochs = 30)

#Preprocess New Image
'''from keras.preprocessing import image
test_image = image.load_img(, target_size)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)'''

# Model deployment
import pickle
pickle.dump(Model, open("flower.pkl", "wb"))