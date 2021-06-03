import cv2 as cv
import numpy as np
import time
import os
import HandTrackingModuleSimple as htm
import math
import shutil

#######################
brushThickness = 15
eraserThickness = 100
########################

folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
drawColor = (255, 0, 255)

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
starCheckTime = 0
isThicknessChange = False
wantToClear = False
isClear = False
isClearCancel = False
startTime = time.time()
actDetectedState = 0
isSavePic = False
isSavePicCancel = False
picIndex = 0
debounceStartTime1 = time.time()
debounceStartTime2 = time.time()
p0x, p0y = 0, 0
p1x, p1y = 0, 0
drawLineStartTime = time.time()
maxDrawlineWaitTime = 1.5
diffPoint = 30
prevState = actDetectedState
picPath = os.path.normpath("./pic")

if os.path.exists(os.path.normpath(picPath)):
    shutil.rmtree(picPath)
    os.makedirs(picPath)
else:
    os.makedirs(picPath)

while True:
    # 1. Import image
    success, img = cap.read()
    img = cv.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)

    if len(lmList) != 0:
        # print(lmList)

        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]


        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        if actDetectedState == 0:
            if actDetectedState != prevState:
                debounceStartTime1 = time.time()
                debounceStartTime2 = time.time()
                starCheckTime = time.time()
                drawLineStartTime = time.time()
            if fingers[0] and fingers[1] and fingers[2] and fingers[3] and fingers[4]:
                if time.time() - debounceStartTime1 > 1.5:
                    # clear the screen
                    isClear = False
                    isClearCancel = False
                    actDetectedState = 1
            else:
                debounceStartTime1 = time.time()
            if fingers[0] == fingers[1] == fingers[2] == fingers[3] == fingers[4] == False:
                if time.time() - debounceStartTime2 > 1.5:
                    # save the pic
                    actDetectedState = 2
                    isSavePic = False
                    isSavePicCancel = False
            else:
                debounceStartTime2 = time.time()
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0
                print("Selection Mode")
                # Checking for the click
                if y1 < 125:
                    if 250 < x1 < 450:
                        header = overlayList[0]
                        drawColor = (255, 0, 255)
                        starCheckTime = time.time()
                    elif 550 < x1 < 750:
                        header = overlayList[1]
                        drawColor = (255, 0, 0)
                        starCheckTime = time.time()
                    elif 850 < x1 < 950:
                        header = overlayList[2]
                        starCheckTime = time.time()
                        drawColor = (0, 255, 0)
                    elif 1050 < x1 < 1200:
                        header = overlayList[3]
                        drawColor = (0, 0, 0)
                        starCheckTime = time.time()
                    elif 0 < x1 < 200:
                        if time.time() - starCheckTime > 3:
                            starCheckTime = time.time()
                            actDetectedState = 4
                cv.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv.FILLED)
            else:
                starCheckTime = time.time()
            if fingers[1] and fingers[2] == False:
                cv.circle(img, (x1, y1), 15, drawColor, cv.FILLED)
                print("Drawing Mode")
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                if drawColor == (0, 0, 0):
                    cv.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                else:
                    cv.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

                xp, yp = x1, y1

                length = math.hypot(x1 - p0x, y1 - p0y)
                if length < diffPoint:
                    d = int(np.interp(time.time()-drawLineStartTime, [0,maxDrawlineWaitTime], [0, 360]))
                    img = cv.ellipse( img, (x1, y1), (30, 30), 0, 0, d, color=(255, 0, 255), thickness = 6)
                    if time.time() - drawLineStartTime > maxDrawlineWaitTime:
                        drawLineStartTime = time.time()
                        actDetectedState = 3
                else:
                    p0x, p0y = x1, y1
                    drawLineStartTime = time.time()

        elif actDetectedState == 1:
            # clear screen
            if isClear == False and isClearCancel == False:
                startTime = time.time()
                cv.putText(img, f'Do you want to clear screen?', (40, 700), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
                if fingers[0] and fingers[1] == fingers[2] == fingers[3] == fingers[4] == False:
                    isClear = True
                    wantToClear = False
                    imgCanvas = np.zeros((720, 1280, 3), np.uint8)
                elif fingers[0] == fingers[1] == fingers[2] == fingers[3] == fingers[4] == False:
                    isClearCancel = True
                    wantToClear = False
            else:
                if isClear == True:
                    cv.putText(img, f'Clear...', (40, 700), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
                    if time.time() - startTime > 1.5:
                        actDetectedState = 0
                if isClearCancel == True:
                    cv.putText(img, f'Cancel.', (40, 700), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
                    if time.time() - startTime > 1.5:
                        actDetectedState = 0


        elif actDetectedState == 2:
            # save picture
            if isSavePic == False and isSavePicCancel == False:
                startTime = time.time()
                cv.putText(img, f'Do you want to save picture?', (40, 700), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
                if fingers[0] and fingers[1] == fingers[2] == fingers[3] == fingers[4] == False:
                    isSavePic = True
                    cv.imwrite((picPath+'/pic%d.jpg')%picIndex,imgCanvas)
                    picIndex = picIndex + 1
                elif fingers[0] == fingers[1] == fingers[2] == fingers[3] == fingers[4] == True:
                    isSavePicCancel = True
            else:
                if isSavePic == True:
                    cv.putText(img, f'Saving...', (40, 700), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
                    if time.time() - startTime > 1.5:
                        actDetectedState = 0
                if isSavePicCancel == True:
                    cv.putText(img, f'Cancel.', (40, 700), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
                    if time.time() - startTime > 1.5:
                        actDetectedState = 0
        elif actDetectedState == 3:
            # draw line
            cv.line(img, (p0x, p0y), (x1, y1), drawColor, brushThickness)
            cv.putText(img, f'Draw a line.', (40, 700), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            length = math.hypot(x1 - p1x, y1 - p1y)
            if length < diffPoint:
                d = int(np.interp(time.time()-drawLineStartTime, [0,maxDrawlineWaitTime], [0, 360]))
                img = cv.ellipse( img, (x1, y1), (30, 30), 0, 0, d, color=(255, 0, 255), thickness = 6)
                if time.time() - drawLineStartTime > maxDrawlineWaitTime:
                    drawLineStartTime = time.time()
                    # cv.line(imgCanvas, (p0x, p0y), (p1x, p1y), drawColor, brushThickness)
                    actDetectedState = 0
            else:
                p1x, p1y = x1, y1
                drawLineStartTime = time.time()
        elif actDetectedState == 4:
            # thickness change
            cv.putText(img, f'Changing Thickness.', (40, 700), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            x3, y3 = lmList[4][1], lmList[4][2]
            x4, y4 = lmList[8][1], lmList[8][2]
            cx, cy = (x3 + x4) // 2, (y3 + y4) // 2

            cv.circle(img, (x3, y3), 15, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x4, y4), 15, (255, 0, 255), cv.FILLED)
            cv.line(img, (x3, y3), (x4, y4), (255, 0, 255), 3)
            cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)

            length = math.hypot(x4 - x3, y4 - y3)
            brushThickness = int(np.interp(length, [50,300], [3, 30]))
            cv.line(img, (700, 700), (800, 700), (255, 0, 255), brushThickness)
            needRefresh = True
            if fingers[1] and fingers[2]:
                # exit condition
                if y1 < 125:
                    if 0 < x1 < 200:
                        needRefresh = False
                        if time.time() - starCheckTime > 3:
                            starCheckTime = time.time()
                            actDetectedState = 0
                            print("Exit")
            if needRefresh == True:
                starCheckTime = time.time()
    else:
        prevState = 255
        actDetectedState = 0
    prevState = actDetectedState
    imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv,cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img,imgInv)
    img = cv.bitwise_or(img,imgCanvas)


    # Setting the header image
    img[0:125, 0:1280] = header
    #img = cv.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv.imshow("Image", img)
    cv.imshow("Canvas", imgCanvas)
    cv.waitKey(1)
