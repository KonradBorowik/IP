import cv2
from imutils import contours
from skimage import measure
import imutils
import numpy as np
import math
from PIL import Image

def SideLength(s1, s2):
    length = math.sqrt((s1[0]-s2[0])**2 + (s1[1]-s2[1])**2)
    return int(length)


def ShortestSide(xy):
    firstApex = xy[0]
    secondApex = xy[1]
    thirdApex = xy[2]

    firstSideLength = SideLength(firstApex, secondApex)
    secondSideLength = SideLength(secondApex, thirdApex)
    thirdSideLength = SideLength(thirdApex, firstApex)

    # first 2 elements are coordinates of the beginning and ending of a side, then its length,
    # last  element are coordinates of the third apex
    firstSide = [firstApex, secondApex, firstSideLength, thirdApex]
    secondSide = [secondApex, thirdApex, secondSideLength, firstApex]
    thirdSide = [thirdApex, firstApex, thirdSideLength, secondApex]

    sides = [firstSide, secondSide, thirdSide]
    sides.sort(key=lambda x: x[2])

    return sides[0]


def MiddlePoint(side):
    x1 = side[0][0]
    x2 = side[1][0]
    y1 = side[0][1]
    y2 = side[1][1]

    if x1 > x2: xMid = x2 + (x1 - x2)/2
    if x2 > x1: xMid = x1 + (x2 - x1)/2

    if y1 > y2: yMid = y2 + (y1 - y2)/2
    if y2 > y1: yMid = y1 + (y2 - y1)/2

    # xMid = x1 + xMid
    # yMid = y1 + yMid

    return [int(xMid), int(yMid)]


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

    xy = []

    # loop over the contours
    for (i, c) in enumerate(cnts):
        # draw the bright spot on the image
        (x, y, w, h) = cv2.boundingRect(c)

        # compute the minimum enclosing circle for each contour
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)

        xy.append([int(cX), int(cY)])

        # draw a circle around desired spots
        cv2.circle(finalImage, (int(cX), int(cY)), int(radius), (0, 0, 255), 2)

        # count each spot
        cv2.putText(finalImage, "#{}".format(i + 1), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    triangleBase = ShortestSide(xy)

    center = MiddlePoint(triangleBase)

    cv2.circle(finalImage, (center[0], center[1]), 9, (0,255,0), 2)

    objectAngle = math.atan2(center[0] - triangleBase[3][0], center[1] - triangleBase[3][1]) * 180 / math.pi

    cv2.arrowedLine(finalImage, center, triangleBase[3], (255,0,0), 2)
    cv2.putText(finalImage, "Angle: {}".format(int(objectAngle)), (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    # show images step by step
    # cv2.imshow("original image", image)
    # cv2.imshow("resized image", resized_image)
    # cv2.imshow("blurred image", blurred)
    # cv2.imshow("blurred graysacle image", gray)
    cv2.imshow("threshold", thresh)
    cv2.imshow("LEDs detected", finalImage)

    cv2.waitKey(0)


Pic1 = cv2.imread(r"C:\Users\konra\PycharmProjects\IP\pictures\same_photo_but_rotated\1.jpg")
# Pic2 = cv2.imread(r"C:\Users\konra\PycharmProjects\IP\pictures\3LEDs_2.jpg")
Pic2 = cv2.imread(r"C:\Users\konra\PycharmProjects\IP\pictures\same_photo_but_rotated\1-1.jpg")
Pic3 = cv2.imread(r"C:\Users\konra\PycharmProjects\IP\pictures\same_photo_but_rotated\1-2.jpg")
Pic4 = cv2.imread(r"C:\Users\konra\PycharmProjects\IP\pictures\same_photo_but_rotated\1-3.jpg")

LedDetector(Pic1)
LedDetector(Pic2)
LedDetector(Pic3)
LedDetector(Pic4)
