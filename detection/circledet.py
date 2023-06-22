import numpy as np
import cv2 as cv

img = cv.imread('testing_dataset/t11.jpg')
img = cv.medianBlur(img, 5)
grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
circles = cv.HoughCircles(grey, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=1, maxRadius=40)

circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    centre = (i[0], i[1])
    print(centre)
    # draw the outer circle
    cv.circle(img, centre, i[2], (0, 255, 0), 2)
    # draw the centre of the circle
    cv.circle(img, centre, 2, (0, 0, 255), 3)

# centres = [(i[0],i[1]) for i in circles [0,:]]

# for centre in centres:
# print(centre)

cv.imshow('detected circles', img)
cv.waitKey(0)
cv.destroyAllWindows()
