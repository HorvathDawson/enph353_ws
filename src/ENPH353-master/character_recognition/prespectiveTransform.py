

# import the necessary packages
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils


def extract_rect(im):
    image = im.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([80, 0, 80])
    upper = np.array([130, 30, 200])
    kernel = np.ones((20, 20), np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.erode(mask, kernel, iterations=4)
    thresh = mask.copy()

    cv2MajorVersion = cv2.__version__.split(".")[0]
    if int(cv2MajorVersion) == 4:
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    # finding contour with max area
    largest = None
    for cnt in contours:
        if largest is None or cv2.contourArea(cnt) > cv2.contourArea(largest):
            largest = cnt
    peri = cv2.arcLength(largest, True)
    appr = cv2.approxPolyDP(largest, 0.02 * peri, True)

    #cv2.drawContours(im, appr, -1, (0,255,0), 3)
    points_list = [[i[0][0], i[0][1]] for i in appr]

    left  = sorted(points_list, key = lambda p: p[0])[0:2]
    right = sorted(points_list, key = lambda p: p[0])[2:4]



    lu = sorted(left, key = lambda p: p[1])[0]
    ld = sorted(left, key = lambda p: p[1])[1]

    ru = sorted(right, key = lambda p: p[1])[0]
    rd = sorted(right, key = lambda p: p[1])[1]



    lu_ = [ (lu[0] + ld[0])/2, (lu[1] + ru[1])/2 ]
    ld_ = [ (lu[0] + ld[0])/2, (ld[1] + rd[1])/2 ]
    ru_ = [ (ru[0] + rd[0])/2, (lu[1] + ru[1])/2 ]
    rd_ = [ (ru[0] + rd[0])/2, (ld[1] + rd[1])/2 ]


    src_pts = np.float32(np.array([lu, ru, rd, ld]))
    dst_pts = np.float32(np.array([lu_, ru_, rd_, ld_]))

    h,w,b = im.shape
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


    imw =  cv2.warpPerspective(im, H, (w, h))

    return imw[lu_[1]:rd_[1], lu_[0]:rd_[0]], imw, lu_, rd_# cropping image
