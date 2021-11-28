from math import atan2, cos, sin, sqrt, pi

import cv2
import imutils
import numpy as np
from skimage import measure
from imutils import contours
from PIL import Image


def drawAxis(img, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)

    ## [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)


def getContours(pts, img):
    ## [pca] (reducing the dimensionality of data)
    # Construct a buffer used by the pca analysis
    # convert contour points into array of contour points
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    ## [pca]

    ## [visualization]
    # Draw the principal components
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    drawAxis(img, cntr, p1, (255, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 0, 0), 5)

    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    ## [visualization]

    # Label with the rotation angle
    label = "  Rotation Angle: " + str(-int(np.rad2deg(angle))) + " degrees"
    textbox = cv2.rectangle(img, (cntr[0] +50, cntr[1] + 50), (cntr[0] + 250, cntr[1] + 10), (255, 255, 255), -1)
    cv2.putText(img, label, (cntr[0]+50, cntr[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return angle


def ledDetector(image):
    # resize
    resized_image = cv2.resize(image, [500, 500])

    # convert to grayscale and apply blur
    blurred = cv2.GaussianBlur(resized_image, (11, 11), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # separate bright spots
    # pixel's value >= 225 set to 255 (white), the rest set to 0 (black)
    thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]

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
        if 30 < numPixels:
            mask = cv2.add(mask, labelMask)

    # find the contours in the mask, then sort them from left to right
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # cnts = contours.sort_contours(cnts)[0]

    # new image to draw circles
    finalImage = resized_image.copy()

    black = np.zeros((500,500,3), dtype='uint8')
    listxy = []

    # loop over the contours
    for (i, c) in enumerate(cnts):
        # draw the bright spot on the image
        (x, y, w, h) = cv2.boundingRect(c)

        # compute the minimum enclosing circle for each contour
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)

        listxy.append((cX, cY))

        # draw a circle around desired spots
        cv2.circle(finalImage, (int(cX), int(cY)), int(radius), (0, 0, 255), 2)

        # count each spot
        cv2.putText(finalImage, "#{}".format(i + 1), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # connecting the apexes
    apexes = np.array(listxy)

    black = cv2.drawContours(black, [apexes.astype(int)], 0, (0,0,255), 1)

    # determining orientation of found triangle
    triangle = black.copy()
    triangle = cv2.fillPoly(triangle, [apexes.astype(int)], color=(0,0,255))
    triangle = np.uint8(triangle)
    triangle_grey = cv2.cvtColor(np.float32(triangle), cv2.COLOR_BGR2GRAY)
    # cv2.imshow('tri grey', triangle_grey)
    # triangle_sides = cv2.threshold(triangle, 50, 255, cv2.THRESH_BINARY)[1]
    _, bw = cv2.threshold(triangle_grey, 50, 255, cv2.THRESH_BINARY)

    bw = np.uint8(bw)

    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for i, c in enumerate(contours):
        # Draw each contour only for visualisation purposes
        cv2.drawContours(finalImage, contours, i, (0, 0, 255), 3)
        # cv2.imshow('asdfasdf', finalImage)
        # Find the orientation of each shape
        getContours(c, finalImage)

    # show images step by step
    # cv2.imshow("original image", image)
    # cv2.imshow("resized image", resized_image)
    # cv2.imshow("blurred image", blurred)
    # cv2.imshow("blurred graysacle image", gray)
    # cv2.imshow("threshold", thresh)
    cv2.imshow("LEDs detected", finalImage)

    # cv2.imshow("L", triangle)

    cv2.waitKey(0)


Pic1 = cv2.imread(r"C:\Users\konra\PycharmProjects\IP\pictures\same_photo_but_rotated\1.jpg")
Pic2 = cv2.imread(r"C:\Users\konra\PycharmProjects\IP\pictures\same_photo_but_rotated\1-1.jpg")
Pic3 = cv2.imread(r"C:\Users\konra\PycharmProjects\IP\pictures\same_photo_but_rotated\1-2.jpg")
Pic4 = cv2.imread(r"C:\Users\konra\PycharmProjects\IP\pictures\same_photo_but_rotated\1-3.jpg")
# Pic5 = cv2.imread(r"C:\Users\konra\PycharmProjects\IP\pictures\Rownoramienny\prosto.jpg")
# Pic6 = cv2.imread(r"C:\Users\konra\PycharmProjects\IP\pictures\Rownoramienny\lewo.jpg")


ledDetector(Pic1)
ledDetector(Pic2)
ledDetector(Pic3)
ledDetector(Pic4)
# ledDetector(Pic5)
# ledDetector(Pic6)
