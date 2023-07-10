import cv2 as cv
import numpy as np
from ultralytics import YOLO


def update_brightness_contrast(brightness=0, contrast=0):
    brightness = cv.getTrackbarPos('Brightness', 'Image')
    contrast = cv.getTrackbarPos('Contrast', 'Image')

    effect = cv.addWeighted(image, 1 + contrast / 127, image, 0, brightness - contrast)
    grey = cv.cvtColor(effect, cv.COLOR_BGR2GRAY)
    grey = cv.medianBlur(grey, 5)


    circles = cv.HoughCircles(grey, cv.HOUGH_GRADIENT, 1, 20,
                              param1=50, param2=30, minRadius=5, maxRadius=30)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv.circle(effect, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv.circle(effect, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv.imshow('Image', effect)
    cv.imshow('edit', grey)


model = YOLO('../runs/detect/train/weights/best.pt')
# source = 'testing_dataset/t10.jpg'
results = model('testing_dataset/k.jpg')
img = cv.imread('testing_dataset/k.jpg')

boxes = results[0].boxes.cpu().numpy()
box = boxes[0]
coord = box.xyxy[0].astype(int)

x1, y1, x2, y2 = coord[1], coord[3], coord[0], coord[2]
image = img[x1:y1, x2:y2]

cv.imshow('Image', image)

cv.createTrackbar('Brightness', 'Image', 0, 2 * 255, update_brightness_contrast)
cv.createTrackbar('Contrast', 'Image', 0, 2 * 127, update_brightness_contrast)

update_brightness_contrast(0, 0)

cv.waitKey()
