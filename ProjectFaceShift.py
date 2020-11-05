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
#anna = cv2.imread('photo_2020-11-04_23-18-10.jpg')
#anna = cv2.cvtColor(anna, cv2.COLOR_BGR2RGB)
#daniel = cv2.imread('photo_2020-11-04_23-18-16.jpg')
#daniel = cv2.cvtColor(daniel, cv2.COLOR_BGR2RGB)
#nicole = cv2.imread('photo_2020-11-04_23-18-19.jpg')
#nicole = cv2.cvtColor(nicole, cv2.COLOR_BGR2RGB)
#nico = cv2.imread('photo_2020-11-04_23-18-22.jpg')
#nico = cv2.cvtColor(nico, cv2.COLOR_BGR2RGB)
#olli = cv2.imread('photo_2020-11-04_23-18-26.jpg')
#olli = cv2.cvtColor(olli, cv2.COLOR_BGR2RGB)
#miri = cv2.imread('photo_2020-11-04_23-18-30.jpg')
#miri = cv2.cvtColor(miri, cv2.COLOR_BGR2RGB)

olli1 = cv2.imread('Olli/20201101_200657.jpg')
olli1 = cv2.cvtColor(olli1, cv2.COLOR_BGR2RGB)
olli2 = cv2.imread('Olli/20201102_081333.jpg')
olli2 = cv2.cvtColor(olli2, cv2.COLOR_BGR2RGB)
olli3 = cv2.imread('Olli/20201103_075426.jpg')
olli3 = cv2.cvtColor(olli3, cv2.COLOR_BGR2RGB)
olli4 = cv2.imread('Olli/20201104_084014.jpg')
olli4 = cv2.cvtColor(olli4, cv2.COLOR_BGR2RGB)

Ollis = [olli1, olli2, olli3, olli4]
refOlli = olli4

############ plotfunctions ###################
def plotImage(image1):
    plt.figure()   
    plt.imshow(image1)
    plt.xticks([]), plt.yticks([])    

def plotFaceWithRect(image1, rect1):
    plt.figure()
    ax1 = plt.subplot(111)
    plt.imshow(image1)
    ax1.add_patch(plt.Rectangle((rect1.left(),rect1.top()),rect1.width(),rect1.height(), edgecolor='r', lw=1.0, fill=False))
    plt.xticks([]), plt.yticks([])   

def plotFaceWithLandmarks(image1, landmarksSource):
    plt.figure()
    ax1 = plt.subplot(111)
    plt.imshow(image1)
    plt.plot(landmarksSource[:,0], landmarksSource[:,1],'.')
    
    plt.xlim(0,image1.shape[1])
    plt.ylim(image1.shape[0],0)
    plt.xticks([]), plt.yticks([])
###############################

def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """

    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    # Transform the coordinates to float
    points1 = np.matrix(points1).astype(np.float64)
    points2 = np.matrix(points2).astype(np.float64)

    # The translation t corresponds to the displacement of the centers of mass.
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)

    # Normalize the mean of the points
    points1 -= c1
    points2 -= c2

    # The scaling corresponds to the ratio between the standard deviation of the points
    s1 = np.std(points1)
    s2 = np.std(points2)

    # Normalize the variance of the points
    points1 /= s1
    points2 /= s2

    # Apply Singular Value decomposition on the correlation matrix of the points
    U, S, Vt = np.linalg.svd(points2.T * points1)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    # Return the affine transformation matrix
    return np.hstack(((s1 / s2) * R,
                     c1.T - (s1 / s2) * R * c2.T))


detector = dlib.get_frontal_face_detector()
faceRects = [detector(img)[0] for img in Ollis]

#for i in range(len(Ollis)):
#    plotFaceWithRect(Ollis[i], faceRects[i])

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
ollilangmarks = []

for i in range(len(Ollis)):
    landmarks = []
    for p in predictor(Ollis[i], faceRects[i]).parts():
        landmarks.append([p.x, p.y])
    landmarks = np.array(landmarks, dtype=np.int) 
    ollilangmarks.append(landmarks)

referencelandmarks = ollilangmarks[3]

#for i in range(len(Ollis)):
#    plotFaceWithLandmarks(Ollis[i], ollilangmarks[i])


#######################################
# 4 - Estimate the transformation between the two faces
#######################################
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)      
transformationmatrices = []

for i in range(len(Ollis)):    
    transformationmatrix = transformation_from_points(ollilangmarks[i][ALIGN_POINTS], referencelandmarks[ALIGN_POINTS])
    transformationmatrices.append(transformationmatrix)

#######################################
# 6 - Warp the source image to the destination
#######################################
#hillary komplett auf merkels groesse
sources_warped = []
for i in range(len(Ollis)):    
    source_warped = cv2.warpAffine(Ollis[i], transformationmatrices[i], (refOlli.shape[1], refOlli.shape[0]), dst = None, borderMode = cv2.BORDER_TRANSPARENT, flags = cv2.WARP_INVERSE_MAP)
    sources_warped.append(source_warped)

for olli in sources_warped:
    plotImage(olli)

plt.show()