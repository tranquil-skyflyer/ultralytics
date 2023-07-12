import math
import numpy as np
import cv2 as cv
from ultralytics import YOLO


def angle(points):
    pt1, pt2 = points
    x1, y1 = pt1
    x2, y2 = pt2
    dx = pt2[0] - pt1[0]
    dy = -(pt2[1] - pt1[1])
    theta = math.atan(dy/dx)
    # theta = math.atan2(pt2[1]-pt1[1], pt2[0]-pt1[0])
    return theta


model = YOLO('../runs/detect/train/weights/best.pt')
results = model('testing_dataset/n.jpg')
img = cv.imread('testing_dataset/n.jpg')

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
grey = cv.medianBlur(grey, 5)
#
# mask = np.zeros(grey.shape[:2], dtype=np.uint8)
# mask[:grey.shape[0]*2//3:, :] = 255
#
# grey = cv.bitwise_and(grey, grey, mask=mask)
#
# cv.imshow('new', grey)

circles = cv.HoughCircles(grey, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=18, maxRadius=40)
centres = []
# circles = np.int16(np.around(circles))
if circles is not None:
    circles = np.int16(np.around(circles))
    # circles = np.round(circles[0, :]).astype(int)
    for i in circles[0, :]:
        centre = (i[0], i[1])
        centres.append((i[0], i[1]))

        # draw the outer circle
        cv.circle(crop, centre, i[2], (0, 255, 0), 2)
        # draw the centre of the circle
        cv.circle(crop, centre, 2, (0, 0, 255), 3)
else:
    print('hi')

print(centres)
cv.imshow('detected circles', crop)

# distances = set()
min_dist = float('inf')
for i in range(len(centres)):
    for j in range(i+1, len(centres)):
        dist = math.dist(centres[i], centres[j])
        if dist < min_dist:
            min_dist = dist
            closest_pts = (centres[i], centres[j])

print(closest_pts)
print(angle(closest_pts))


# print(min(distances))

# circles = np.round(circles[0, :]).astype(int)
# c1 = (circles[0][0], circles[0][1])
# c2 = (circles[1][0], circles[1][1])
# if circles[2] is not None:
#     c3 = (circles[2][0], circles[2][1])
# vector = (circles[1][0] - circles[0][0], -(circles[1][1]-circles[0][1]))
# print(angle(vector))

# d1 = math.dist(c1, c2)
# d2 = math.dist(c1, c3)
# d3 = math.dist(c2, c3)



# cx1, cy1 = circles[0][0], circles[0][1]
# cx2, cy2 = circles[1][0], circles[1][1]
# cx3, cy3 = circles[2][0], circles[2][1]
#
# d1 = (cx2 - cx1, (cy1 - cy2))
# d2 = (cx3 - cx1, (cy1 - cy3))
# d3 = (cx3 - cx2, (cy2 - cy3))
# distances = {d1, d2, d3}
# print(min(distances))
# a = angle(min(distances))


# print(a)


cv.waitKey(0)
cv.destroyAllWindows()
