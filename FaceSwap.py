# Copyright (c) 2015 Matthew Earl
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
This is the code behind the Switching Eds blog post:
    
    http://matthewearl.github.io/2015/07/28/switching-eds-with-python/

The code is available at:

    https://github.com/matthewearl/faceswap/blob/master/faceswap.py

To run the script you'll need to install dlib (http://dlib.net) including its
Python bindings, and OpenCV. 

You'll also need to obtain and unzip the trained model from sourceforge:

    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
"""
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
source = cv2.imread('photo_2020-11-04_23-18-10.jpg')
source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
destination = cv2.imread('photo_2020-11-04_23-18-30.jpg')
destination = cv2.cvtColor(destination, cv2.COLOR_BGR2RGB)

def plotImage(image):
    plt.figure()
    plt.subplot(111)
    plt.imshow(image)
    plt.xticks([]), plt.yticks([])

def plotImages(image1, image2):
    plt.figure()
    plt.subplot(121)
    plt.imshow(image1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122)
    plt.imshow(image2)
    plt.xticks([]), plt.yticks([])

#plotImages(source, destination)
#######################################
# 2 - Viola-Jones Face detector on the two images
#######################################

detector = dlib.get_frontal_face_detector()
faceRectSource = detector(source)[0]
faceRectDest = detector(destination)[0]

#######################################
def plotFacesWithRects(image1, rect1, image2, rect2):
    plt.figure()
    ax1 = plt.subplot(121)
    plt.imshow(image1)
    ax1.add_patch(plt.Rectangle((rect1.left(),rect1.top()),rect1.width(),rect1.height(), edgecolor='r', lw=1.0, fill=False))
    plt.xticks([]), plt.yticks([])
    
    ax1 = plt.subplot(122)
    plt.imshow(image2)
    ax1.add_patch(plt.Rectangle((rect2.left(),rect2.top()),rect2.width(),rect2.height(), edgecolor='r', lw=1.0, fill=False))
    plt.xticks([]), plt.yticks([])

plotFacesWithRects(source, faceRectSource, destination, faceRectDest)

#######################################
# 3 - Get the facial landmarks using dlib
#######################################

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
source_landmarks = []
for p in predictor(source, faceRectSource).parts():
    source_landmarks.append([p.x, p.y])
source_landmarks = np.array(source_landmarks, dtype=np.int) 

destination_landmarks = []
for p in predictor(destination, faceRectDest).parts():
    destination_landmarks.append([p.x, p.y])
destination_landmarks = np.array(destination_landmarks, dtype=np.int) 

#######################################
def plotFacesWithLandmarks(image1, landmarksSource, image2, landmarksDestination):
    plt.figure()
    ax1 = plt.subplot(121)
    plt.imshow(image1)
    plt.plot(landmarksSource[:,0], landmarksSource[:,1],'.')
    
    plt.xlim(0,image1.shape[1])
    plt.ylim(image1.shape[0],0)
    plt.xticks([]), plt.yticks([])
   
    ax1 = plt.subplot(122)
    plt.imshow(image2)
    plt.plot(landmarksSource[:,0], landmarksDestination[:,1],'.')
    
    plt.xlim(0,image2.shape[1])
    plt.ylim(image2.shape[0],0)
    plt.xticks([]), plt.yticks([])

#plotFacesWithLandmarks(source, source_landmarks, destination, destination_landmarks)

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

transformationmatrix = transformation_from_points(source_landmarks[ALIGN_POINTS], destination_landmarks[ALIGN_POINTS])


#######################################
# 5 - Extract a mask in the two images
#######################################

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

def get_face_mask(img, landmarks):
    "Extracts a mask on an image around the important regions: eyebrows, eyes, nose, mouth."
    # Create an empty mask
    mask = np.zeros(img.shape[:2], dtype=np.float64)

    # Compute the mask by computing the convex hull.
    for group in OVERLAY_POINTS:
        points = cv2.convexHull(landmarks[group])
        cv2.fillConvexPoly(mask, points, color=1)

    # Transform the mask into an image
    mask = np.array([mask, mask, mask]).transpose((1, 2, 0))

    # Blur the mask
    mask = (cv2.GaussianBlur(mask, (11, 11), 0) > 0) * 1.0
    mask = cv2.GaussianBlur(mask, (11, 11), 0)

    return mask

mask_source = get_face_mask(source, source_landmarks)
mask_destination = get_face_mask(destination, destination_landmarks)

def plotMasks(mask1, mask2):
    plt.figure()
    ax1 = plt.subplot(121)
    plt.imshow(mask1)
    
    plt.xlim(0,mask1.shape[1])
    plt.ylim(mask1.shape[0],0)
    plt.xticks([]), plt.yticks([])
    
    ax2 = plt.subplot(122)
    plt.imshow(mask2)
    
    plt.xlim(0,mask2.shape[1])
    plt.ylim(mask2.shape[0],0)
    plt.xticks([]), plt.yticks([])
   
source_masked = (np.uint8)(mask_source * source) #hillary
destnation_masked = (np.uint8) (mask_destination * destination) #angela
#plotMasks(source_masked, destnation_masked)
#######################################
# 6 - Warp the source image to the destination
#######################################
#hillary komplett auf merkels groesse
source_warped = cv2.warpAffine(source, transformationmatrix, (destination.shape[1], destination.shape[0]), dst = None, borderMode = cv2.BORDER_TRANSPARENT, flags = cv2.WARP_INVERSE_MAP)

#plotImages(source_warped, destination)
#######################################
# 7 - Warp the source mask to the destination
#######################################

warped_mask = cv2.warpAffine(mask_source, transformationmatrix,(destination.shape[1], destination.shape[0]), dst = None, borderMode = cv2.BORDER_TRANSPARENT, flags = cv2.WARP_INVERSE_MAP)
combined_mask = np.max([mask_destination, warped_mask], axis = 0) #hillarys maske in merkels gesicht

hillarysAusschnitt = (np.uint8)(source_warped * warped_mask)
merkelOhneAusschnitt = (np.uint8)(destination*(1-warped_mask))


#plotImages(hillarysAusschnitt, merkelOhneAusschnitt)

#######################################
# 8 - Add the warped source image to the destination using the mask
#######################################

endeVomLied = hillarysAusschnitt + merkelOhneAusschnitt
plotImage(endeVomLied)
#######################################
# 9 - Correct the colors
#######################################

#dest, source, destlandmarks
def correct_colours(original, target, landmarks):
    "RGB color scaling correction"

    # Compute the size of the Gaussian filter by measuring the distance between the eyes
    blur_amount = np.linalg.norm(
                      np.mean(landmarks[LEFT_EYE_POINTS], axis=0) -
                      np.mean(landmarks[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1

    # Blur the two images
    original_blur = cv2.GaussianBlur(original, (blur_amount, blur_amount), 0)
    target_blur = cv2.GaussianBlur(target, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    target_blur += (128 * (target_blur <= 1.0)).astype(target_blur.dtype)

    # Compute the color-corrected image
    return (target.astype(np.float64) * original_blur.astype(np.float64) / target_blur.astype(np.float64))


hillaryAngeglichen = correct_colours(destination, source_warped, destination_landmarks)
hillarysAusschnitt = (np.uint8)(source_warped * warped_mask)
#plotImages(hillaryAngeglichen, hillaryAngeglichen)
#hillaryAngeglichen = (cv2.normalize(hillaryAngeglichen, None,0.0,255.0, cv2.NORM_MINMAX)).astype(np.uint8)

#plotImages(hillaryAngeglichen, hillaryAngeglichen)

#plotImages(hillaryAngeglichen.astype(np.uint8), hillarysAusschnitt)

plt.show()

