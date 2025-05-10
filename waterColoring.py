"""
Image Processing Example with openCV
Water Coloring

"""

import cv2 as cv
import numpy as np


def reduceValHSV(val, factor):
    return factor * (val // factor)


def reduceColorsHSV(img, factor):
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv_img)
    cv.imshow('h image', h)


    size = img.shape
    for i in range(size[0]):
        for j in range(size[1]):
            h[i, j] = reduceValHSV(h[i, j], factor)
            s[i, j] = int(s[i, j] * 0.75)
            v[i, j] = min(int(v[i, j] * 1.5), 255)

    hsv_img = cv.merge([h, s, v])
    rgb_img = cv.cvtColor(hsv_img, cv.COLOR_HSV2BGR)
    return rgb_img


def generate_edges(img):
    grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return cv.Canny(grey_img, 10, 120)


# ------------------------------------
img = cv.imread('./PlayGroundDetection/assets/playground1.png', cv.IMREAD_COLOR)
cv.imshow('input', img)

size = img.shape
print(size)

kernel = np.ones((5, 5), np.float32)/25
smoothed_img = cv.filter2D(img, -1, kernel)

edge_img = generate_edges(smoothed_img)
# cv.imshow('edge image', edge_img)

reduced_color_img = reduceColorsHSV(smoothed_img, 10);
# cv.imshow('reduced colors', reduced_color_img)

for i in range(size[0]):
    for j in range(size[1]):
        if edge_img[i, j] != 0:
            reduced_color_img[i, j][0] = 0
            reduced_color_img[i, j][1] = 0
            reduced_color_img[i, j][2] = 0

cv.imshow('water color image', reduced_color_img)


cv.waitKey(0)
