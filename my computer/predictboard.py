import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from split_image import split_image
import array as arr
import os
import subprocess
import time

# for save file
#9 7
numsequence = '7'

#cmd = "raspistill -o ./img/base/image" + str(numsequence) + ".jpg"
#subprocess.call(cmd, shell=True)
#time.sleep(1)
# load image

print('put your img at ./img/base/image '+ str(numsequence) + '.jpg')
img = cv2.imread('./img/base/image' + str(numsequence) + '.jpg')

# apply median blur, 15 means it's smoothing image 15x15 pixels
blur = cv2.medianBlur(img, 15)

# convert to hsv
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

# color definition
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([130, 255, 255])

# red color mask (sort of thresholding, actually segmentation)
mask = cv2.inRange(hsv, lower_blue, upper_blue)

connectivity = 4
# Perform the operation
output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
# Get the results

num_labels = output[0] - 1

centroids = output[3][1:]

# Traitement sur les points => enlever les duplication
copions = [];
list_centre = [];

if num_labels > 4:
    print(str(num_labels) + " dots found")
    for j in range(num_labels):
        # Verifie si le temp existe dans copions
        temp_list = [[centroids[j][0], centroids[j][1]]]
        if ([temp_list[0][0], temp_list[0][1]] in copions):
            continue;

        # ajoute ses copions a la liste des copions
        for i in range(num_labels):
            if ((temp_list[0][0] < centroids[i][0] + 100 and temp_list[0][0] > centroids[i][0] - 100) and (
                    temp_list[0][1] < centroids[i][1] + 100 and temp_list[0][1] > centroids[i][1] - 100) and (
                    temp_list[0][0] != centroids[i][0])):
                print("repetition")
                copions.append([centroids[i][0], centroids[i][1]])

        list_centre.append([temp_list[0][0], temp_list[0][1]])

elif num_labels == 4:
    print("4 found")
    for i in range(num_labels):
        # transfere a list_centre
        list_centre.append([centroids[i][0], centroids[i][1]])
else:
    print(str(num_labels) + " dots found")
    for i in range(num_labels):
        # transfere a list_centre
        list_centre.append([centroids[i][0], centroids[i][1]])

print('array of dot center coordinates:', centroids)
print('copions:', copions)
print('cleaned:', list_centre)

# Put pixels of the chess corners: top left, top right, bottom right, bottom left.
list_somme_coord = [];
for i in range(len(list_centre)):
    list_somme_coord.append(list_centre[i][0] + list_centre[i][1]);

print("List sommes coord :")
print(list_somme_coord)

# find top right and bottom left corners coord indexes,top left => min somme of coord, bot right => max somme of coord
list_index_notUsed = [];
for i in range(len(list_centre)):
    if (i not in [list_somme_coord.index(min(list_somme_coord)), list_somme_coord.index(max(list_somme_coord))]):
        list_index_notUsed.append(i)

print("index not used" + str(list_index_notUsed))
bot_left_index = 0;
top_right_index = 0;
if (list_centre[list_index_notUsed[0]][0] < list_centre[list_index_notUsed[1]][0]):
    # we found bottom left by comparing x position (smallest)
    bot_left_index = list_index_notUsed[0]
    top_right_index = list_index_notUsed[1]
else:
    bot_left_index = list_index_notUsed[1]
    top_right_index = list_index_notUsed[0]

cornerPoints = np.array([[list_centre[list_somme_coord.index(min(list_somme_coord))][0],
                          list_centre[list_somme_coord.index(min(list_somme_coord))][1]],
                         [list_centre[top_right_index][0], list_centre[top_right_index][1]],
                         [list_centre[list_somme_coord.index(max(list_somme_coord))][0],
                          list_centre[list_somme_coord.index(max(list_somme_coord))][1]],
                         [list_centre[bot_left_index][0], list_centre[bot_left_index][1]]
                         ], dtype='float32')


# Find base of the rectangle given by the chess corners
base = np.linalg.norm(cornerPoints[1] - cornerPoints[0])

# Height has 8 squares and base has 8 squares.
height = base / 8 * 8

# Define new corner points from base and height of the rectangle
new_cornerPoints = np.array([[0, 0], [int(base), 0], [int(base), int(height)], [0, int(height)]], dtype='float32')

# Calculate matrix to transform the perspective of the image
M = cv2.getPerspectiveTransform(cornerPoints, new_cornerPoints)

new_image = cv2.warpPerspective(img, M, (int(base), int(height)))


# Function to get data points in the new perspective from points in the image
def calculate_newPoints(points, M):
    new_points = np.einsum('kl, ...l->...k', M,
                           np.concatenate([points, np.broadcast_to(1, (*points.shape[:-1], 1))], axis=-1))
    return new_points[..., :2] / new_points[..., 2][..., None]




# Paint new data points in red
#cv2.imshow("result", new_image)
cv2.imwrite('./img/total/' + str(numsequence) + 'board.jpg', new_image)

# Image splitting

nRows = 8
# Number of columns
mCols = 8

# Reading image
img = cv2.imread('./img/total/' + str(numsequence) + 'board.jpg')


# Dimensions of the image
sizeX = img.shape[1]
sizeY = img.shape[0]

print("Image shape :")
print(img.shape)

for i in range(0, nRows):
    for j in range(0, mCols):
        roi = img[int(i * sizeY / nRows):int(i * sizeY / nRows + sizeY / nRows), int(j * sizeX / mCols):int(j * sizeX / mCols + sizeX / mCols)]
        cv2.imwrite("./img/part/" + str(numsequence) + 'test_' + str(i) + str(j) + '.jpg', roi)



#establish the ia with weights


model = keras.models.load_model('./model/mymodel7')
model.summary()

image_size = (200, 200)
myvect = ['bcheval', 'bfou', 'bpion', 'breine', 'broi', 'btour', 'ncheval', 'nfou', 'npion', 'nreine', 'nroi','ntour', 'vide']
#myvect = [ 'blanc' , 'noir','vide']
chessmatrix = []
for i in range(0, nRows):
    row =[]
    for j in range(0, mCols):
        img = keras.preprocessing.image.load_img(
            "./img/part/" + str(numsequence) + 'test_' + str(int(i)) + str(int(j)) + '.jpg', target_size=image_size
        )

        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)

        maxi = max(predictions[0])
      #  print(str(maxi) +'is max')
        pos = 0
        for k in predictions[0]:
            #print(str(k)+'to comp ')
            pos = pos + 1
            #print('ok at pos '+str(pos)+'**/'+str (myvect[int(pos-1)]))
            if (k == maxi):

                row.append( str (myvect[pos-1]) )
    chessmatrix.append(row)

print("results")
for row in chessmatrix:
    print("-----------------------------------------------------------------------------------------")
    print(row)






