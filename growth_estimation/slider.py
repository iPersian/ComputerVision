import cv2
import numpy as np
from tkinter.filedialog import askopenfilenames


"""
Slider application for automatically extract masks
"""

paths = askopenfilenames()
images = []


cv2.namedWindow("TrackedBars")
cv2.resizeWindow("TrackedBars", 640, 360)


for path in paths:
    img = cv2.imread(path)
    img = cv2.resize(img, (0, 0), None, 1, 1, interpolation=cv2.INTER_CUBIC)
    images.append(img)


def on_trackbar(val):
    hue_min = cv2.getTrackbarPos("Hue Min", "TrackedBars")
    hue_max = cv2.getTrackbarPos("Hue Max", "TrackedBars")
    sat_min = cv2.getTrackbarPos("Sat Min", "TrackedBars")
    sat_max = cv2.getTrackbarPos("Sat Max", "TrackedBars")
    val_min = cv2.getTrackbarPos("Val Min", "TrackedBars")
    val_max = cv2.getTrackbarPos("Val Max", "TrackedBars")
    d = cv2.getTrackbarPos("d", "TrackedBars")
    sigma_color = cv2.getTrackbarPos("Sigma Color", "TrackedBars")
    sigma_space = cv2.getTrackbarPos("Sigma Space", "TrackedBars")

    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])

    res = []
    for img in images:
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(imgHSV, lower, upper)
        mask_fil = cv2.bilateralFilter(mask, d, sigma_color, sigma_space)
        numpy_horizontal = np.hstack((img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), cv2.cvtColor(mask_fil, cv2.COLOR_GRAY2BGR)))
        res.append(numpy_horizontal)

    cv2.imshow('Numpy Horizontal', np.vstack(res))


cv2.createTrackbar("Hue Min", "TrackedBars", 30, 179, on_trackbar)
cv2.createTrackbar("Hue Max", "TrackedBars", 50, 179, on_trackbar)
cv2.createTrackbar("Sat Min", "TrackedBars", 87, 255, on_trackbar)
cv2.createTrackbar("Sat Max", "TrackedBars", 255, 255, on_trackbar)
cv2.createTrackbar("Val Min", "TrackedBars", 86, 255, on_trackbar)
cv2.createTrackbar("Val Max", "TrackedBars", 255, 255, on_trackbar)
cv2.createTrackbar("d", "TrackedBars", 20, 50, on_trackbar)
cv2.createTrackbar("Sigma Color", "TrackedBars", 121, 255, on_trackbar)
cv2.createTrackbar("Sigma Space", "TrackedBars", 34, 255, on_trackbar)


# Show some stuff
on_trackbar(0)
# Wait until user press some key
cv2.waitKey()