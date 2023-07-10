import math
# import os
import glob
import numpy as np
import cv2 as cv
from ultralytics import YOLO


def angle(xy):
    a = xy[1] / xy[0]
    theta = math.atan(a)
    return theta


model = YOLO('../runs/detect/train/weights/best.pt')  # load yolo model
source = 'testing_dataset/*.jpg'  # image source (need to change None for path)
# results = model(source)  # predict on dataset -- inside or out of for loop?


# for loop to loop thru each file
for file in glob.glob(source):
    results = model(file)
    img = cv.imread(file)
    boxes = results[0].boxes.cpu().numpy()
    box = boxes[0]
    coord = box.xyxy[0].astype(int)

    # img = cv.imread(os.path.join(source, file))  # will be i-th image
    # crop
    x1, y1, x2, y2 = coord[1], coord[3], coord[0], coord[2]
    crop = img[x1:y1, x2:y2]

    # resize images if needed
    h, w, c = crop.shape
    if w < 180 and h < 180:
        while w < 180 and h < 180:
            crop = cv.resize(crop, None, fx=1.2, fy=1.2)
            h, w, c = crop.shape
    else:
        while w > 180 and h > 180:
            crop = cv.resize(crop, None, fx=0.8, fy=0.8)
            h, w, c = crop.shape



    # detect circles
    crop = cv.medianBlur(crop, 5)
    grey = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    grey = cv.convertScaleAbs(grey, alpha=4.25, beta=25)

    mask = np.zeros(grey.shape[:2], dtype=np.uint8)
    mask[:grey.shape[0] * 2 // 3:, :] = 255  # mask lower third of image

    grey = cv.bitwise_and(grey, grey, mask=mask)

    circles = cv.HoughCircles(grey, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=15, maxRadius=30)


    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            centre = (i[0], i[1])
            print(centre)
            # draw the outer circle
            cv.circle(crop, centre, i[2], (0, 255, 0), 2)
            # draw the centre of the circle
            cv.circle(crop, centre, 2, (0, 0, 255), 3)

        circles = np.round(circles[0, :]).astype(int)
        centre = (circles[0][0], circles[0][1])
        centre1 = (circles[1][0], circles[1][1])
        vector = (circles[1][0] - circles[0][0], -(circles[1][1] - circles[0][1]))
        print(angle(vector))
    # else:
    #     print(file + " has no circles")

    cv.imshow('detected circles', crop)
    cv.waitKey(0)
    cv.destroyAllWindows()

