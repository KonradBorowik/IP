import cv2
import numpy as np
from imutils import contours
from skimage import measure
import imutils


cap = cv2.VideoCapture(1)

while True:
    timer = cv2.getTickCount()
    success, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)[1]

    fps = cv2.getTickFrequency()/(cv2.getTickCount() - timer)
    cv2.putText(thresh, str(int(fps)), (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.imshow("camera 1", thresh)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break