import cv2

closer = cv2.imread(r"C:\Users\konra\PycharmProjects\LearningGitHub\pictures\closer.jpg")
closer_resized = cv2.resize(closer, [500,500], interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(closer_resized, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 11), 0)

(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
cv2.circle(closer_resized, maxLoc, 21, (255, 0, 0), 2)

# cv2.imshow("closer resized", gray)
# cv2.waitKey(0)

cv2.imshow("LED detected", closer_resized)
cv2.waitKey(0)

