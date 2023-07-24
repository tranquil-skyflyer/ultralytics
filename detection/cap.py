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
    grey = cv.convertScaleAbs(grey, alpha=4.5, beta=20)

    centres = []
    circles = cv.HoughCircles(grey, cv.HOUGH_GRADIENT, 1.2, grey.shape[0] / 4, param1=50, param2=30, minRadius=10,
                              maxRadius=35)
    if circles is not None:
        circles = np.int16(np.around(circles))
        for i in circles[0, :]:
            centres.append((i[0], i[1]))

    return centres


def distances(coords):
    min_dist = float('inf')
    closest_pts = ()
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = math.dist(coords[i], coords[j])
            if dist < min_dist:
                min_dist = dist
                closest_pts = (coords[i], coords[j])
    return closest_pts


def angle(points):
    pt1, pt2 = points
    dx = pt2[0] - pt1[0]
    dy = -(pt2[1] - pt1[1])
    theta = math.atan(dy / dx)
    return theta  # add tolerances??


# def is_mid_centre(mid_x, shape):
#     centre_x = shape / 2
#     tolerance = shape / 24
#     return abs(mid_x - centre_x) <= tolerance


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

    colour_image = np.asanyarray(colour_frame.get_data())
    print("Image captured")
    results = model.predict(colour_image, conf=0.75)
    annotated_frame = results[0].plot()
    boxes = results[0].boxes.cpu().numpy()

    cv.namedWindow('RealSense', cv.WINDOW_AUTOSIZE)
    cv.imshow('Realsense', annotated_frame)
    cv.waitKey(1)

    count += 1
    if count % 2 == 0:
        continue

    if len(boxes) != 0:
        box = boxes[0]
        coord = box.xyxy[0].astype(int)
        x1, y1, x2, y2 = coord[1], coord[3], coord[0], coord[2]
        crop = colour_image[x1:y1, x2:y2]
        ctr = circle(crop)
        print(ctr)
        if len(ctr) > 1:
            print('success!')
            d = distances(ctr)
            ang = round(angle(d), 4)
            # midpoint_x = (d[0][0] + d[1][0]) // 2
            # img_width = colour_image.shape[1]
            # aligned = is_mid_centre(midpoint_x, img_width)  # consider adding tolerance-specifc function
            annotated_frame = cv.putText(annotated_frame, str(ang), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1,
                                         (0, 0, 255), 1)
    else:
        print('no detection :(')

    cv.namedWindow('RealSense', cv.WINDOW_AUTOSIZE)
    cv.imshow('Realsense', annotated_frame)
    cv.waitKey(1)

# TODO:
#  need to figure out how to stream all frames but only detect every other frame
#  add threshold for objectiveness and circular
#  use ratio instead of radius
#  add tolerances for angles???
