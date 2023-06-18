import numpy as np
import cv2 as cv
from ultralytics import YOLO


model = YOLO('runs/detect/train9/weights/best.pt')  # load yolo model
# Predict with the model -- INSERT crop
source = None  # image source (need to change None for path)
results = model('source')  # predict on dataset -- inside or out of for loop?

# will need to create loop in order go thru all predicted images
# loop would start here
boxes = results[0].boxes.cpu().numpy()
box = boxes[0]
coord = box.xyxy[0].astype(int)
img = cv.imread(None) # will be n-th image


# crop -- no for loop needed since only one box
x1, y1, x2, y2 = coord[1], coord[3], coord[0], coord[2]
crop = img[x1:y1, x2:y2]


# detect circles
crop = cv.imread(crop, 0)
crop = cv.medianBlur(crop, 5)
cimg = cv.cvtColor(crop,cv.COLOR_GRAY2BGR)
circles = cv.HoughCircles(crop,cv.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the centre of the circle
    cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv.imshow('detected circles',cimg)
cv.waitKey(0)
cv.destroyAllWindows()

# for loop would end here
