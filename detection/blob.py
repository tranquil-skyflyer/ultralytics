import numpy as np
import cv2 as cv

from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')
results = model('testing_dataset/t10.jpg')
img = cv.imread('testing_dataset/t10.jpg', cv.IMREAD_GRAYSCALE)

boxes = results[0].boxes.cpu().numpy()
box = boxes[0]
coord = box.xyxy[0].astype(int)

x1, y1, x2, y2 = coord[1], coord[3], coord[0], coord[2]
crop = img[x1:y1, x2:y2]
crop = cv.medianBlur(crop, 5)
crop = cv.convertScaleAbs(crop, alpha=2.25, beta=0)

params = cv.SimpleBlobDetector_Params()

# Disable convexity parameter
params.filterByConvexity = False

# Area parameters
params.filterByArea = True
params.minArea = 5

# Circularity parameters
params.filterByCircularity = True
params.minCircularity = 0.9

# Inertial parameters
params.filterByInertia = True
params.minInertiaRatio = 0.5

detector = cv.SimpleBlobDetector_create(params)
keypoints = detector.detect(crop)

blobs = cv.drawKeypoints(crop, keypoints, np.zeros((1,1)), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow('Keypoints', blobs)
cv.waitKey(0)
cv.destroyAllWindows()
