
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

    # # regions of interest
    sorted_ctrs = sorted(ctrs, key=lambda x: cv2.contourArea(x))[:4]
    sorted_ctrs = sorted(sorted_ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    if len(sorted_ctrs) < 4:
        return ValueError("less than 4 values found")

    licenseStr = ""
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        image = resized[y - 10:y + h + 10, x - 10:x + w +10]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.addWeighted(image, 5, cv2.blur(image, (150, 150)), -4, 128)
        licenseStr += str(model.predict(image)[0])
    return licenseStr

def find_ParkingSpot(img, model):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 0])
    upper = np.array([150, 50, 50])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.array([[1,1],[1,1]],dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    cv2MajorVersion = cv2.__version__.split(".")[0]
    # check for contours on thresh
    if int(cv2MajorVersion) == 4:
        ctrs, hier = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        im2, ctrs, hier = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sorted_ctrs = sorted(ctrs, key=lambda x: cv2.contourArea(x))[:2]
    sorted_ctrs = sorted(sorted_ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    if len(sorted_ctrs) < 2:
        return ValueError("less than 2 values found")

    parkingspotStr = ""

    x, y, w, h = cv2.boundingRect(sorted_ctrs[1])
    if h > w:
        w = h
    image = img[y:y + h, x:x + w]

    image = cv2.addWeighted(image, 5, cv2.blur(image, (150, 150)), -4, 128)
    prediction = model.predict(image)

    for i in range(len(prediction)):
        if str.isalpha(prediction[i]):
            continue
        return prediction[i]
    return str(0)
