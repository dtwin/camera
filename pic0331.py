import cv2

img = cv2.imread('/home/pi/camera/0331_1.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
img1, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


#for i in range(len(contours)):
   # (x, y), radius = cv2.minEnclosingCircle(contours[i])
   # center = (int(x), int(y))
   # radius = int(radius)
   # img = cv2.circle(img, center, radius, (0, 255, 0), 2)


cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
img = img.resize(1280,960)

cv2.imshow("img", img)
cv2.imwrite("/home/pi/camera/1.jpg",img)
cv2.waitKey(0)

