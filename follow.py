import cv2

cap = cv2.VideoCapture(1)

while True:
    timer = cv2.getTickCount()
    success, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fps = cv2.getTickFrequency()/(cv2.getTickCount() - timer)
    cv2.putText(gray, str(fps), (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.imshow("camera 1", gray)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break