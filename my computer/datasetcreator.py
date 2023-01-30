import cv2
import numpy as np
from split_image import split_image
import array as arr
import os
import subprocess
import time



# for save file
numsequence ='15'



#cap = cv2.VideoCapture(0)
#success , img = cap.read()
#cv2.imwrite('./img/global/base/'+str(numsequence)+'board.jpg',img)
cmd = "raspistill -o ./img/base/image"+str(numsequence)+".jpg"
subprocess.call(cmd, shell=True)
time.sleep(1)
#load image
img = cv2.imread('./img/base/image'+str(numsequence)+'.jpg')

#apply median blur, 15 means it's smoothing image 15x15 pixels
blur = cv2.medianBlur(img,15)

#convert to hsv
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

#color definition
#red_lower = np.array([161, 155, 84])
#red_upper = np.array([179, 255, 255])

#lower_blue = np.array([110,50,50])
#upper_blue = np.array([130,255,255])
#lower_blue= np.array([100,70,60])
#upper_blue = np.array([130,120,150])
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([130, 255, 255])
#red color mask (sort of thresholding, actually segmentation)
mask = cv2.inRange(hsv, lower_blue, upper_blue)

connectivity = 4
# Perform the operation
output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
# Get the results

num_labels = output[0]-1

centroids = output[3][1:]

#Traitement sur les points => enlever les duplication
copions=[];
list_centre=[];

if num_labels > 4:
    print(str(num_labels)+" dots found")
    for j in range(num_labels):
        #Verifie si le temp existe dans copions
        temp_list = [ [centroids[j][0],centroids[j][1]]  ]
        if([temp_list[0][0],temp_list[0][1]] in copions):
            continue;

        #ajoute ses copions a la liste des copions
        for i in range(num_labels):
            if((temp_list[0][0]<centroids[i][0]+100 and temp_list[0][0]>centroids[i][0]-100) and (temp_list[0][1]<centroids[i][1]+100 and temp_list[0][1]>centroids[i][1]-100) and (temp_list[0][0]!=centroids[i][0])):
                #print("temp_list[0][0] "+str(temp_list[0][0])+"centroids[i][0]+100 "+str(centroids[i][0]+100)+"temp_list[0][0] "+str(temp_list[0][0])+"centroids[i][0] "+str(centroids[i][0]) )
                print("repetition")
                copions.append([centroids[i][0],centroids[i][1]])
        
        list_centre.append([temp_list[0][0],temp_list[0][1]])

elif num_labels==4:
    print("4 found")
    for i in range(num_labels):
        #transfere a list_centre
        list_centre.append([centroids[i][0],centroids[i][1]])
else:
    print(str(num_labels)+" dots found")
    for i in range(num_labels):
        #transfere a list_centre
        list_centre.append([centroids[i][0],centroids[i][1]])

print ('array of dot center coordinates:',centroids)
print ('copions:',copions)
print ('cleaned:',list_centre)


#Put pixels of the chess corners: top left, top right, bottom right, bottom left.
list_somme_coord=[];
for i in range(len(list_centre)):
    list_somme_coord.append(list_centre[i][0]+list_centre[i][1]);

print("List sommes coord :")
print(list_somme_coord)

#find top right and bottom left corners coord indexes,top left => min somme of coord, bot right => max somme of coord
list_index_notUsed =[];
for i in range(len(list_centre)):
    if(i not in [list_somme_coord.index(min(list_somme_coord)),list_somme_coord.index(max(list_somme_coord))]):
        list_index_notUsed.append(i)

print("index not used"+str(list_index_notUsed))
bot_left_index=0;
top_right_index=0;
if(list_centre[list_index_notUsed[0]][0] < list_centre[list_index_notUsed[1]][0]):
    #we found bottom left by comparing x position (smallest)
    bot_left_index=list_index_notUsed[0]
    top_right_index=list_index_notUsed[1]
else:
    bot_left_index=list_index_notUsed[1]
    top_right_index=list_index_notUsed[0]

cornerPoints = np.array([[ list_centre[list_somme_coord.index(min(list_somme_coord))][0]  , list_centre[list_somme_coord.index(min(list_somme_coord))][1]   ],
 [ list_centre[top_right_index][0] , list_centre[top_right_index][1]],
 [list_centre[list_somme_coord.index(max(list_somme_coord))][0]  , list_centre[list_somme_coord.index(max(list_somme_coord))][1]],
 [list_centre[bot_left_index][0] , list_centre[bot_left_index][1]]
 ], dtype='float32')

'''
cornerPoints = np.array([[  428.47 , 1073.114   ],
 [ 2517.31     ,    1066.56],
 [ 2555.51 , 3105.45],
 [442.280 , 3142.8795]

 ], dtype='float32')

 cornerPoints = np.array([[ list_centre[list_somme_coord.index(min(list_somme_coord))][0]  , list_centre[list_somme_coord.index(min(list_somme_coord))][1]   ],
 [ 2517.31     ,    1066.56],
 [list_centre[list_somme_coord.index(max(list_somme_coord))][0]  , list_centre[list_somme_coord.index(max(list_somme_coord))][1]],
 [442.280 , 3142.8795]
 ], dtype='float32')

'''

#Find base of the rectangle given by the chess corners
base = np.linalg.norm(cornerPoints[1] - cornerPoints[0] )

#Height has 8 squares and base has 8 squares.
height = base/8*8

#Define new corner points from base and height of the rectangle
new_cornerPoints = np.array([[0, 0], [int(base), 0], [int(base), int(height)], [0, int(height)]], dtype='float32')

#Calculate matrix to transform the perspective of the image
M = cv2.getPerspectiveTransform(cornerPoints, new_cornerPoints)

new_image = cv2.warpPerspective(img, M, (int(base), int(height)))

#Function to get data points in the new perspective from points in the image
def calculate_newPoints(points, M):
    new_points = np.einsum('kl, ...l->...k', M,  np.concatenate([points, np.broadcast_to(1, (*points.shape[:-1], 1)) ], axis = -1) )
    return new_points[...,:2] / new_points[...,2][...,None]

#new_points = calculate_newPoints(points, M)

#Paint new data points in red
#for i in range(len(new_points)):
    #cv2.circle(new_image, tuple(new_points[i].astype('int64')), radius=0, color=(0, 0, 255), thickness=5)

#cv2.imwrite('new_undistorted.png', new_image)
cv2.imshow("result", new_image)
cv2.imwrite('./img/total/'+str(numsequence)+'board.jpg',new_image)


#Image splitting

nRows = 8
# Number of columns
mCols = 8

# Reading image
img = cv2.imread('./img/total/'+str(numsequence)+'board.jpg')
#img = cv2.resize(img,(1000,1000))

#print img

#cv2.imshow('image',img)

# Dimensions of the image
sizeX = img.shape[1]
sizeY = img.shape[0]

print("Image shape :")
print(img.shape)


for i in range(0,nRows):
    for j in range(0, mCols):
        roi = img[   int(i*sizeY/nRows):int(i*sizeY/nRows + sizeY/nRows) ,int(j*sizeX/mCols):int(j*sizeX/mCols + sizeX/mCols)    ]
        #cv2.imshow('rois'+str(i)+str(j), roi)
        #dirname = 'c:/Users/ralib/Source/M28/Q1/Gestion_Systeme_Vision/ProjetJeuSociete/Projet/Brouillon/Final/stockImageTest'
        #os.mkdir(dirname)
        cv2.imwrite("./img/part/1"+str(numsequence)+'test_'+str(i)+str(j)+'.jpg', roi)
