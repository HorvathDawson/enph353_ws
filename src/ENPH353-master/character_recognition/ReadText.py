
import numpy as np
import cv2
import os
from prespectiveTransform import extract_rect

def find_license(img, model):
    imgCopy = img.copy()
    scale_percent = 1000 # percent of original size
    width = int(imgCopy.shape[1] * scale_percent / 100)
    height = int(imgCopy.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    lower = np.array([115, 115, 90])
    upper = np.array([127, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)

    cv2MajorVersion = cv2.__version__.split(".")[0]
    # check for contours on thresh
    if int(cv2MajorVersion) == 4:
        ctrs, hier = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        im2, ctrs, hier = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    # regions of interest
    ROI = np.array([[0, 0, 0, 0, 0]])

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        b = np.array([[w * h, x, y, w, h]])
        ROI = np.concatenate((ROI, b), axis=0)
        # Getting ROI
        roi = mask[y:y + h, x:x + w].copy()

    # sort by max area
    ROI = -1 * ROI
    ROI = -1 * ROI[ROI[:, 0].argsort()]
    # remove non characters
    ROI = ROI[0:4, :]
    # sort by x values
    ROI = ROI[ROI[:, 1].argsort()]
    augimg = img.copy()

    licenseStr = ""
    for A, x, y, w, h in ROI:
        image = resized[y - 10:y + h + 10, x - 10:x + w +10]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.addWeighted(image, 5, cv2.blur(image, (150, 150)), -4, 128)
        licenseStr += str(model.predict(image)[0])
    return licenseStr

def find_ParkingSpot(img, model):
    imgCopy = img.copy()
    scale_percent = 1000 # percent of original size
    width = int(imgCopy.shape[1] * scale_percent / 100)
    height = int(imgCopy.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    lower = np.array([115, 115, 90])
    upper = np.array([127, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)

    cv2MajorVersion = cv2.__version__.split(".")[0]
    # check for contours on thresh
    if int(cv2MajorVersion) == 4:
        ctrs, hier = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        im2, ctrs, hier = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    # regions of interest
    ROI = np.array([[0, 0, 0, 0, 0]])

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        b = np.array([[w * h, x, y, w, h]])
        ROI = np.concatenate((ROI, b), axis=0)
        # Getting ROI
        roi = mask[y:y + h, x:x + w].copy()

    # sort by max area
    ROI = -1 * ROI
    ROI = -1 * ROI[ROI[:, 0].argsort()]
    # remove non characters
    ROI = ROI[0:4, :]
    # sort by x values
    ROI = ROI[ROI[:, 1].argsort()]
    augimg = img.copy()

    licenseStr = ""
    for A, x, y, w, h in ROI:
        image = resized[y - 10:y + h + 10, x - 10:x + w +10]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.addWeighted(image, 5, cv2.blur(image, (150, 150)), -4, 128)
        licenseStr += str(model.predict(image)[0])
    return licenseStr
