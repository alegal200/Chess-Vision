import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pathlib



model = keras.models.load_model('./model/mymodel7')
model.summary()

image_size = (200, 200)


img = keras.preprocessing.image.load_img(
    "test/j.jpg", target_size=image_size
)

img_array = keras.preprocessing.image.img_to_array(img)

img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)


print('le score est ')
#print( predictions[0] )
vect =[]
for i in predictions[0]:
   vect.append(int(i*100))

print("----")
print(vect)
myvect = ['bcheval', 'bfou', 'bpion', 'breine', 'broi', 'btour', 'ncheval', 'nfou', 'npion', 'nreine', 'nroi', 'ntour', 'vide']
#myvect = [ 'blanc' , 'noir','vide']
print(myvect)

for i in range(3):
    print('-'+str(myvect[i])+'->'+str(vect[i]))

#
#
#
#
#
#
#
#
#
#



















#print("package  Imorted")
#
#cap = cv2.VideoCaoture(0)
#cap.set(3,640)
#cap.set(4,480)


#while True :
#    success , img = cap.read()
#    cv2.imshow("Video",img)
#    if cv2.waitKey(1) && 0xFF == ord('q'):
#        break
