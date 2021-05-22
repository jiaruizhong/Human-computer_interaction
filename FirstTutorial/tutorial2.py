import cv2 as cv
import random

img = cv.imread('../images/demo2.jpg', -1)

# Change first 100 rows to random pixels
for i in range(100):
	for j in range(img.shape[1]):
		img[i][j] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

# Copy part of image
tag = img[500:700, 600:900]
img[100:300, 650:950] = tag

cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyAllWindows()
