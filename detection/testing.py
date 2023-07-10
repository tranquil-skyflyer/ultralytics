import math
import numpy as np
import cv2 as cv
from ultralytics import YOLO


def angle(xy):
    a = xy[1]/xy[0]
    theta = math.atan(a)
    return theta


model = YOLO('../runs/detect/train/weights/best.pt')
results = model('testing_dataset/b.jpg')  # issues: not detecting circles for t13.jpg
img = cv.imread('testing_dataset/b.jpg')

boxes = results[0].boxes.cpu().numpy()
box = boxes[0]
coord = box.xyxy[0].astype(int)

x1, y1, x2, y2 = coord[1], coord[3], coord[0], coord[2]
crop = img[x1:y1, x2:y2]


h, w, c = crop.shape
if w < 180 and h < 180:
    while w < 180 and h < 180:
        crop = cv.resize(crop, None, fx=1.2, fy=1.2)
        h, w, c = crop.shape

# crop = cv.medianBlur(crop, 5)
grey = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
# grey = cv.addWeighted(grey, 1 + 200 / 127, grey, 0, 225 - 200)
grey = cv.convertScaleAbs(grey, alpha=4.25, beta=10)
grey = cv.medianBlur(grey, 7)

mask = np.zeros(grey.shape[:2], dtype=np.uint8)
mask[:grey.shape[0]*2//3:, :] = 255

grey = cv.bitwise_and(grey, grey, mask=mask)

# cv.imshow('new', grey)

circles = cv.HoughCircles(grey, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=18, maxRadius=40)

# circles = np.int16(np.around(circles))
if circles is not None:
    circles = np.int16(np.around(circles))
    # circles = np.round(circles[0, :]).astype(int)
    for i in circles[0, :]:
        centre = (i[0], i[1])
        print(centre)
        # draw the outer circle
        cv.circle(crop, centre, i[2], (0, 255, 0), 2)
        # draw the centre of the circle
        cv.circle(crop, centre, 2, (0, 0, 255), 3)
else:
    print("hi")


circles = np.round(circles[0, :]).astype(int)
# centre = (circles[0][0], circles[0][1])
# centre1 = (circles[1][0], circles[1][1])
vector = (circles[1][0] - circles[0][0], -(circles[1][1]-circles[0][1]))
print(angle(vector))

# cv.imshow('detected circles', crop)
# cv.waitKey(0)
# cv.destroyAllWindows()
