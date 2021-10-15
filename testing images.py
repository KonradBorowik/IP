import cv2


def leddetector(image):
    resized_image = cv2.resize(image, [500, 500], interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 11), 0)

    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blurred)
    cv2.circle(resized_image, maxLoc, 21, (255, 0, 0), 2)

    # cv2.imshow("LED detected", gray)
    cv2.imshow("LED detected", resized_image)
    cv2.waitKey(0)


img = cv2.imread(r"C:\Users\konra\PycharmProjects\LearningGitHub\pictures\close.jpg")

leddetector(img)
