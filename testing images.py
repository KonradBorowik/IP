import cv2


def leddetector(image):
    image_resized = cv2.resize(image, [500, 500], interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 11), 0)

    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    cv2.circle(image_resized, maxLoc, 21, (255, 0, 0), 2)

    # cv2.imshow("image resized", gray)
    # cv2.waitKey(0)

    cv2.imshow("LED detected", image_resized)
    cv2.waitKey(0)

closer = cv2.imread(r"C:\Users\konra\PycharmProjects\LearningGitHub\pictures\closer.jpg")

leddetector(closer)