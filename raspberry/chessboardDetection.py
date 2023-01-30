import cv2
import numpy as np
from split_image import split_image
import array as arr
import os
import subprocess
import time

cmd = "raspistill -o ./img/test.jpg"
subprocess.call(cmd, shell=True)
time.sleep(1)

#load image
img = cv2.imread('./img/test.jpg')

#apply median blur, 15 means it's smoothing image 15x15 pixels
blur = cv2.medianBlur(img,15)
#convert to hsv
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
rhsv = cv2.resize(hsv, (960, 540))  
cv2.imshow("hsv",rhsv)
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([130, 255, 255])
#red color mask (sort of thresholding, actually segmentation)
mask = cv2.inRange(hsv, lower_blue, upper_blue)
rmask = cv2.resize(mask, (960, 540))  
cv2.imshow("mask",rmask)

while True :
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
print('fin')