# opencv-python
import cv2 as cv

img = cv.imread('../images/download.jpg')
img = cv.resize(img, (0, 0), fx=0.5, fy=0.5)
img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)

cv.imwrite("new_img.jpg", img)

cv.imshow('Camera Demo', img)
cv.waitKey(0)
cv.destroyAllWindows()
