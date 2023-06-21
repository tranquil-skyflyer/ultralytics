import cv2 as cv


def BrightnessContrast(brightness=0):
    # getTrackbarPos returns the current
    # position of the specified trackbar.
    brightness = cv.getTrackbarPos('Brightness', 'GEEK')

    contrast = cv.getTrackbarPos('Contrast', 'GEEK')

    effect = controller(img, brightness, contrast)

    # The function imshow displays an image
    # in the specified window
    cv.imshow('Effect', effect)


def controller(img, brightness=255, contrast=127):
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))

    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))

    if brightness != 0:

        if brightness > 0:

            shadow = brightness

            max = 255

        else:

            shadow = 0
            max = 255 + brightness

        al_pha = (max - shadow) / 255
        ga_mma = shadow

        # The function addWeighted calculates
        # the weighted sum of two arrays
        # cal = cv.addWeighted(img, al_pha, img, 0, ga_mma)


    else:
        cal = img

    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)

        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv.addWeighted(cal, Alpha, cal, 0, Gamma)

    # putText renders the specified text string in the image.
    cv.putText(cal, 'B:{},C:{}'.format(brightness, contrast), (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return cal


if __name__ == '__main__':
    # The function imread loads an image
    # from the specified file and returns it.
    original = cv.imread('testing_dataset/t3.jpg')

    # Making another copy of an image.
    img = original.copy()

    # The function namedWindow creates a
    # window that can be used as a placeholder
    # for images.
    cv.namedWindow('GEEK')

    # The function imshow displays an
    # image in the specified window.
    cv.imshow('GEEK', original)

    # createTrackbar(trackbarName,
    # windowName, value, count, onChange)
    # Brightness range -255 to 255
    cv.createTrackbar('Brightness', 'GEEK', 255, 2 * 255, BrightnessContrast)

    # Contrast range -127 to 127
    cv.createTrackbar('Contrast', 'GEEK', 127, 2 * 127, BrightnessContrast)

    BrightnessContrast(0)

# The function waitKey waits for
# a key event infinitely or for delay
# milliseconds, when it is positive.
cv.waitKey(0)
