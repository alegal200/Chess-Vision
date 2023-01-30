import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import pathlib
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.models import Model
import os
print("package  Imorted")


######################################
#
# put your dataset here ./dataset
#
# it's generate a model in ./model
#
#
#####################################








print('debut')
#todo change val 13
#parametres :
batch_size = 3
img_height = 200
img_width = 200
num_classes = 13
epochevalue = 250
data_dir = pathlib.Path('./dataset')
print('->')


#chemin du dataset
print(data_dir)
print(os.path.abspath(data_dir))

# un-use here but usefull
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("vertical"),
        layers.RandomRotation(0.1),
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomContrast(factor=0.1),

    ])


#données d entraibement
train_data = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  )

#données de validation

val_data = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = val_data.class_names
print(class_names)



model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),

    layers.Conv2D(128,4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256,activation='relu'),
    layers.Dense(512 ,activation='relu'),
    layers.Dense(256,activation='relu'),
    layers.Dense(128,activation='relu'),
    layers.Dense(64,activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

#   layers.Conv2D(16,4, activation='relu'),
#   layers.MaxPooling2D(),



model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'],)

logdir="logs"

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=1, write_images=logdir,
                                                   embeddings_data=train_data)

model.fit(
    train_data,
  validation_data=val_data,
  epochs=epochevalue,
  callbacks=[tensorboard_callback]
)

model.summary()

model.save('./model/mymodel9')

























#model = tf.keras.Sequential([
#    data_augmentation,
#    layers.experimental.preprocessing.Rescaling(1./255),
#    layers.Conv2D(256,4, activation='relu'),
#    layers.MaxPooling2D(),
#    layers.Conv2D(128,4, activation='relu'),
#    layers.MaxPooling2D(),
#    layers.Conv2D(64,4, activation='relu'),
#    layers.MaxPooling2D(),
#    layers.Conv2D(32,4, activation='relu'),
    #layers.MaxPooling2D(),
    #layers.Conv2D(16,4, activation='relu'),
#    layers.MaxPooling2D(),
#    layers.Flatten(),
#    layers.Dense(64,activation='relu'),
#    layers.Dense(64,activation='relu'),
#    layers.Dense(num_classes, activation='softmax')
#])




































print('fin gg ca compile bg ')

#cap = cv2.VideoCapture(0)
#print("--------")
# prepare object points

#while True :
#
#    success , img = cap.read()
#    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    (retval ,mimg2) =cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
#    contours, hierarchy = cv2.findContours(mimg2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#    mimg2 = cv2.drawContours(mimg2, contours, -1, (0, 255, 75), 2)

#    if success:

#        cv2.imshow("im", mimg2)


#    if cv2.waitKey(1) & 0xFF ==ord('q'):
#        break