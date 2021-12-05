import cv2
from imutils import contours
from skimage import measure
import imutils
import numpy as np
import math
from PIL import Image
from collections import OrderedDict


def length(s1, s2):
    length = math.sqrt((s1[0]-s2[0])**2 + (s1[1]-s2[1])**2)

    return length


def GetShortestSide(apexes):
    apex1 = apexes[0]
    apex2 = apexes[1]
    apex3 = apexes[2]

    length1 = length(apex1, apex2)
    length2 = length(apex2, apex3)
    length3 = length(apex3, apex1)

    # SORT THIS SHIT OUT

    side1 = OrderedDict(apex1[0], apex1[1], apex2[0], apex2[1], int(length1))
    side2 = OrderedDict(apex2[0], apex2[1], apex3[0], apex3[1], int(length2))
    side3 = OrderedDict(apex3[0], apex3[1], apex1[0], apex1[1], int(length3))
    print('aaa')


def LedDetector(image):
    # resize
    resized_image = cv2.resize(image, [500, 500])

    # convert to grayscale and apply blur
    blurred = cv2.GaussianBlur(resized_image, (11, 11), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # separate bright spots
    # pixel's value >= 225 set to 255 (white), the rest set to 0 (black)
    thresh = cv2.threshold(gray, 224, 255, cv2.THRESH_BINARY)[1]

    # perform a connected component analysis on the thresholded image
    labels = measure.label(thresh, connectivity=2, background=0)

    # initialize a mask to store only large enough (unique) components
    mask = np.zeros(thresh.shape, dtype="uint8")

    # loop over the unique components to separate actual light sources from reflections
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue

        # otherwise, construct the label mask and count the number of pixels in the mask
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if 50 < numPixels:
            mask = cv2.add(mask, labelMask)

    # find the contours in the mask, then sort them from left to right
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]

    # new image to draw circles
    finalImage = resized_image.copy()

    apexes = []

    # loop over the contours
    for (i, c) in enumerate(cnts):
        # draw the bright spot on the image
        (x, y, w, h) = cv2.boundingRect(c)

        # compute the minimum enclosing circle for each contour
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)

        apexes.append([x, y])

        # draw a circle around desired spots
        cv2.circle(finalImage, (int(cX), int(cY)), int(radius), (0, 0, 255), 2)

        # count each spot
        cv2.putText(finalImage, "#{}".format(i + 1), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    GetShortestSide(apexes)

    # show images step by step
    # cv2.imshow("original image", image)
    # cv2.imshow("resized image", resized_image)
    # cv2.imshow("blurred image", blurred)
    # cv2.imshow("blurred graysacle image", gray)
    cv2.imshow("threshold", thresh)
    cv2.imshow("LEDs detected", finalImage)

    cv2.waitKey(0)


Pic1 = cv2.imread(r"C:\Users\konra\PycharmProjects\IP\pictures\Rownoramienny\lewo.jpg")
Pic2 = cv2.imread(r"C:\Users\konra\PycharmProjects\IP\pictures\3LEDs_2.jpg")

LedDetector(Pic1)