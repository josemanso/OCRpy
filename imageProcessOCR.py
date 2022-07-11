#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 12:06:33 2022

@author: josemo
"""

import cv2
import pytesseract
#import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("/home/josemo/Documentos/imagenes/image.jpg")

h,w,c =image.shape
print(f'Image shape: {h}H x {w}W x {c}C ')

# get grayscaleimage
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)

# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #retrun cv2.threshold(image, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

# dilation
def dilate(image):
    kernel = np.ones((5,5), np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)

# erosion
def erode (image):
    kernel = np.ones((5,5), np.uint8)
    return cv2.erode(image, kernel, iterations =1)

# openong - erosion followed by dilation
def opening (image):
    kernel = np.ones((5,5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# canny edge detection
def canny (image):
    return cv2.Canny(image, 100, 200)

# skew correction
def deskew (image):
    coords = np.column_stack(np.where(image >0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90+angle)
    else:
        angle = -angle
    (h, w)= image.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.wrapAffine(image, M, (w,h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated

# template matching
def match_template (image, template):
    return cv2.matchTemplate (image, template, cv2.TM_CCOEFF_NORMED)


# prepocesing
gray = get_grayscale(image)
thresh = thresholding(gray)
opening = opening(gray)
canny = canny(gray)

# show
# show the image, provide window name first
cv2.imshow('image window', image)
cv2.imshow('gray image', gray)
cv2.imshow('thresh image', thresh)
cv2.imshow('opening image', opening)
cv2.imshow('canny image', canny)
# add wait key. window waits until user presses a key
cv2.waitKey(0)
# and finally destroy/close all open windows
cv2.destroyAllWindows()

# print text
print(" gray scale")
print(pytesseract.image_to_string(gray))
print(" thresh")
print(pytesseract.image_to_string(thresh))



