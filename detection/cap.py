import math
import pyrealsense2 as rs
import numpy as np
import cv2 as cv
from ultralytics import YOLO


def circle(img):
    h, w, c = img.shape
    if w < 180 and h < 180:
        while w < 180 and h < 180:
            img = cv.resize(img, None, fx=1.2, fy=1.2)
            h, w, c = img.shape

    img = cv.medianBlur(img, 5)
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    grey = cv.convertScaleAbs(grey, alpha=4.25, beta=20)
    mask = np.zeros(grey.shape[:2], dtype=np.uint8)
    mask[:grey.shape[0] * 2 // 3:, :] = 255

    grey = cv.bitwise_and(grey, grey, mask=mask)
    circles = cv.HoughCircles(grey, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=18, maxRadius=40)
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        # print(len(circles))


        if len(circles) == 2:
            cx1 = circles[0][0]
            cy1 = circles[0][1]
            cx2 = circles[1][0]
            cy2 = circles[1][1]
            vector = (cx2 - cx1, -(cy2 - cy1))
            return vector
    else:
        return None


def angle(xy):
    a = xy[1] / xy[0]
    theta = math.atan(a)
    return theta


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

model = YOLO('../runs/detect/train/weights/best.pt')
count = 0

while True:
    frames = pipeline.wait_for_frames()
    colour_frame = frames.get_color_frame()
    if not colour_frame:
        continue

    count += 1
    if count % 2 == 0:
        continue

    colour_image = np.asanyarray(colour_frame.get_data())
    print("Image captured")
    results = model.predict(colour_image, conf=0.75)
    annotated_frame = results[0].plot()
    boxes = results[0].boxes.cpu().numpy()

    if len(boxes) != 0:
        box = boxes[0]
        coord = box.xyxy[0].astype(int)
        x1, y1, x2, y2 = coord[1], coord[3], coord[0], coord[2]
        crop = colour_image[x1:y1, x2:y2]
        vec = circle(crop)
        if vec is not None:
            print('success!')
            ang = round(angle(vec), 4)
            annotated_frame = cv.putText(annotated_frame, str(ang), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    else:
        print('no detection :(')

    cv.namedWindow('RealSense', cv.WINDOW_AUTOSIZE)
    cv.imshow('Realsense', annotated_frame)
    cv.waitKey(1)



# TODO:
#  need to figure out how to stream all frames but only detect every other frame
#  add threshold for objectiveness and circular
#  use ratio instead of radius
