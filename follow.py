import cv2
import numpy as np
from imutils import contours
from skimage import measure
import imutils
import math


def SideLength(s1, s2):
    length = math.sqrt((s1[0]-s2[0])**2 + (s1[1]-s2[1])**2)
    return int(length)


def ShortestSide(xy):
    first_apex = xy[0]
    second_apex = xy[1]
    third_apex = xy[2]

    first_side_length = SideLength(first_apex, second_apex)
    second_side_length = SideLength(second_apex, third_apex)
    third_side_length = SideLength(third_apex, first_apex)

    # first 2 elements are coordinates of the beginning and ending of a side, then its length,
    # last  element are coordinates of the third apex
    first_side = [first_apex, second_apex, first_side_length, third_apex]
    second_side = [second_apex, third_apex, second_side_length, first_apex]
    third_side = [third_apex, first_apex, third_side_length, second_apex]

    sides = [first_side, second_side, third_side]
    sides.sort(key=lambda x: x[2])

    return sides[0]


def MiddlePoint(side):
    x1 = side[0][0]
    x2 = side[1][0]
    y1 = side[0][1]
    y2 = side[1][1]

    if x1 > x2:
        if y1 > y2:
            x_mid = x2 + (x1 - x2)/2
            y_mid = y2 + (y1 - y2) / 2
        else:
            x_mid = x2 + (x1 - x2) / 2
            y_mid = y1 + (y2 - y1) / 2

        return [int(x_mid), int(y_mid)]

    if x2 > x1:
        if y2 > y1:
            x_mid = x1 + (x2 - x1)/2
            y_mid = y1 + (y2 - y1) / 2
        else:
            y_mid = y2 + (y1 - y2) / 2
            x_mid = x1 + (x2 - x1)/2

        return [int(x_mid), int(y_mid)]


def DestinationAngle(object_center, destination_point):
    angle = math.atan2(object_center[0] - destination_point[0], object_center[1] - destination_point[1]) * 180 / math.pi
    return angle

def CheckAngle(object_angle, destionation_angle):
    if object_angle < destionation_angle:
        print("left")
    if object_angle > destionation_angle:
        print("right")
    if object_angle == destionation_angle:
        print("go forward")


route = [[250, 250], [200, 200], [300, 200], [300, 300], [200, 300]]
cap = cv2.VideoCapture(1)
x = 0

while True:
    timer = cv2.getTickCount()
    success, img = cap.read()

    resized_image = cv2.resize(img, [500, 500])

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
        label_mask = np.zeros(thresh.shape, dtype="uint8")
        label_mask[labels == label] = 255
        num_pixels = cv2.countNonZero(label_mask)

        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if 20 < num_pixels:
            mask = cv2.add(mask, label_mask)

    # find the contours in the mask, then sort them from left to right
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]

    # new image to draw circles
    final_image = resized_image.copy()

    xy = []

    # loop over the contours
    for (i, c) in enumerate(cnts):
        # draw the bright spot on the image
        (x, y, w, h) = cv2.boundingRect(c)

        # compute the minimum enclosing circle for each contour
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)

        xy.append([int(cX), int(cY)])

        # draw a circle around desired spots
        cv2.circle(final_image, (int(cX), int(cY)), int(radius), (0, 0, 255), 2)

        # count each spot
        cv2.putText(final_image, "#{}".format(i + 1), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    if len(xy) < 3:
        cv2.putText(final_image, "{X}", (220, 220), cv2.FONT_HERSHEY_SIMPLEX, 15, (0,0,255), 4)
        continue

    triangleBase = ShortestSide(xy)

    center = MiddlePoint(triangleBase)

    if center:
        cv2.circle(final_image, (center[0], center[1]), 7, (0,255,0), 4)
    else:
        continue

    object_angle = math.atan2(center[0] - triangleBase[3][0], center[1] - triangleBase[3][1]) * 180 / math.pi

    cv2.arrowedLine(final_image, center, triangleBase[3], (255, 0, 0), 2)
    cv2.putText(final_image, "Angle: {}".format(int(object_angle)), (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    destination_angle = DestinationAngle(center, route[x])

    CheckAngle(object_angle, destination_angle)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)[1]

    fps = cv2.getTickFrequency()/(cv2.getTickCount() - timer)
    cv2.putText(final_image, str(int(fps)), (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.imshow("camera 1", final_image)

    if center == route[x]:
        x += 1

    if cv2.waitKey(1) & 0xff == ord('q'):
        break