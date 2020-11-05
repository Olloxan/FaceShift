from __future__ import print_function
import cv2
import dlib
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

#######################################
# 1 - Open the two images and convert them to RGB
#######################################
anna = cv2.imread('photo_2020-11-04_23-18-10.jpg')
anna = cv2.cvtColor(anna, cv2.COLOR_BGR2RGB)
daniel = cv2.imread('photo_2020-11-04_23-18-16.jpg')
daniel = cv2.cvtColor(daniel, cv2.COLOR_BGR2RGB)
nicole = cv2.imread('photo_2020-11-04_23-18-19.jpg')
nicole = cv2.cvtColor(nicole, cv2.COLOR_BGR2RGB)
nico = cv2.imread('photo_2020-11-04_23-18-22.jpg')
nico = cv2.cvtColor(nico, cv2.COLOR_BGR2RGB)
olli = cv2.imread('photo_2020-11-04_23-18-26.jpg')
olli = cv2.cvtColor(olli, cv2.COLOR_BGR2RGB)
miri = cv2.imread('photo_2020-11-04_23-18-30.jpg')
miri = cv2.cvtColor(miri, cv2.COLOR_BGR2RGB)

currentImg = nicole

def plotImages(image1):
    plt.figure()   
    plt.imshow(image1)
    plt.xticks([]), plt.yticks([])    


detector = dlib.get_frontal_face_detector()
faceRect = detector(currentImg)[0]


def plotFacesWithRects(image1, rect1):
    plt.figure()
    ax1 = plt.subplot(111)
    plt.imshow(image1)
    ax1.add_patch(plt.Rectangle((rect1.left(),rect1.top()),rect1.width(),rect1.height(), edgecolor='r', lw=1.0, fill=False))
    plt.xticks([]), plt.yticks([])   
    
plotFacesWithRects(currentImg, faceRect)

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
landmarks = []
for p in predictor(currentImg, faceRect).parts():
    landmarks.append([p.x, p.y])
landmarks = np.array(landmarks, dtype=np.int) 

#######################################
def plotFacesWithLandmarks(image1, landmarksSource):
    plt.figure()
    ax1 = plt.subplot(111)
    plt.imshow(image1)
    plt.plot(landmarksSource[:,0], landmarksSource[:,1],'.')
    
    plt.xlim(0,image1.shape[1])
    plt.ylim(image1.shape[0],0)
    plt.xticks([]), plt.yticks([])
      

plotFacesWithLandmarks(currentImg, landmarks)


plt.show()