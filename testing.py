import numpy as np
import cv2 as cv
from ultralytics import YOLO

model = YOLO('runs/detect/train9/weights/best.pt')
results = model('testing.jpg')
img = cv.imread('testing.jpg')

boxes = results[0].boxes.cpu().numpy()
box = boxes[0]
coord = box.xyxy[0].astype(int)

x1, y1, x2, y2 = coord[1], coord[3], coord[0], coord[2]
crop = img[x1:y1, x2:y2]


crop = cv.imread(crop, 0)
crop = cv.medianBlur(crop, 5)
cimg = cv.cvtColor(crop, cv.COLOR_GRAY2BGR)
circles = cv.HoughCircles(crop, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # draw the outer circle
    cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the centre of the circle
    cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

cv.imshow('detected circles', cimg)
cv.waitKey(0)
cv.destroyAllWindows()


#  File "testing.py", line 17, in <module>
#    crop = cv.imread(crop, 0)
#TypeError: Can't convert object to 'str' for 'filename'
