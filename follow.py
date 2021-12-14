import cv2
import numpy as np
from imutils import contours
from skimage import measure
import imutils


cap = cv2.VideoCapture(1)

success, img = cap.read()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)[1]

labels = measure.label(thresh, connectivity=2, background=0)

mask = np.zeros(thresh.shape, dtype="uint8")

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

# loop over the contours
for (i, c) in enumerate(cnts):
    # draw the bright spot on the image
    (x, y, w, h) = cv2.boundingRect(c)

    # compute the minimum enclosing circle for each contour
    ((cX, cY), radius) = cv2.minEnclosingCircle(c)

    # draw a circle around desired spots
    cv2.circle(img, (int(cX), int(cY)), int(radius), (0, 0, 255), 2)

    # count each spot
    cv2.putText(img, "#{}".format(i + 1), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
 #TODO tutaj trzeba jakos wziac i przekazac te punkty z osobna do sledzenia

while True:
    timer = cv2.getTickCount()
    success, img = cap.read()

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fps = cv2.getTickFrequency()/(cv2.getTickCount() - timer)
    cv2.putText(img, str(int(fps)), (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.imshow("camera 1", img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break