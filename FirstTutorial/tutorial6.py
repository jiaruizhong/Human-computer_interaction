import numpy as np
import cv2 as cv

img = cv.imread('../images/chessboard.jpg')
img = cv.resize(img, (0, 0), fx=0.75, fy=0.75)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

corners = cv.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int0(corners)

for corner in corners:
	x, y = corner.ravel()
	cv.circle(img, (x, y), 5, (255, 0, 0), -1)

for i in range(len(corners)):
	for j in range(i + 1, len(corners)):
		corner1 = tuple(corners[i][0])
		corner2 = tuple(corners[j][0])
		color = tuple(map(lambda x: int(x), np.random.randint(0, 255, size=3)))
		cv.line(img, corner1, corner2, color, 1)

cv.imshow('Frame', img)
cv.waitKey(0)
cv.destroyAllWindows()
