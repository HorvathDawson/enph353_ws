import cv2
import numpy as np
from datetime import datetime

def find_lines(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    (thresh, processed) = cv2.threshold(
        blur_gray, 210, 255, cv2.THRESH_BINARY)
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5, 5), np.uint8)
    processed = cv2.dilate(processed, kernel, iterations=2)
    processed = cv2.erode(processed, kernel, iterations=2)
    return processed

def find_Red(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 50, 50])
    upper = np.array([5, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def find_Truck(im):


    return im

def find_Cars(im):
    img = im[:,600:].copy()
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    lower = np.array([118, 50, 50])
    upper = np.array([130, 255, 255])
    kernel = np.ones((20, 20), np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def find_Grass(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    lower = np.array([70, 50, 50])
    upper = np.array([75, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    processed = cv2.dilate(mask, kernel, iterations=2)
    processed = cv2.erode(mask, kernel, iterations=2)
    return processed

def find_roads(im):
    img = im.copy()
    # img[-250:,:] = 0 finds it perpendicular
    # img[-150:,:] = 0 works
    img[-20:,:] = 0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 70])  # 50
    upper = np.array([0, 0, 110])  # 100
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    kernel = np.ones((2, 8), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=7)

    cv2MajorVersion = cv2.__version__.split(".")[0]
    if int(cv2MajorVersion) == 4:
        ctrs, hier = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
	im2, ctrs, hier = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    im = im.copy()
    x, y, w, h = cv2.boundingRect(sorted_ctrs[0])
    # cv2.rectangle(im,(x,y),(x+w,y+h),155,5)
    return w, im

def hugh_lines(im):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    bw = find_lines(im)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(bw,(kernel_size, kernel_size),0)
    # blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 25  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 30  # maximum gap in pixels between connectable line segments
    line_image = np.copy(im) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            if (y2-y1)/float(x2-x1) < 0:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

    return line_image
def COM(im):
    # calculate moments of binary image
    M = cv2.moments(im)
    # calculate x,y coordinate of center
    if M["m00"] > 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX = 0
        cY = 0
    return cX, cY
def filter_cars(im):
    bwimg = find_Cars(im)
    carFound = False
    cv2MajorVersion = cv2.__version__.split(".")[0]
    # check for contours on thresh
    if int(cv2MajorVersion) == 4:
        ctrs, hier = cv2.findContours(bwimg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
	im2, ctrs, hier = cv2.findContours(bwimg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    im = im.copy()

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        b = np.array([[w * h, x, y, w, h]])
    	if((w*h > ((1200*700)/12))):
    	    carFound = True
            cv2.rectangle(im,(x,y),(x+w,y+h),155,5)

            if(im is not None):
                imCrop = im[y:y+h,x:x+w]
                cv2.imwrite("/home/dawson/enph353_ws/src/control_pkg/licensePlateProcess/licensePlateImages/" + str(datetime.now().time()) +  ".png",imCrop)

    return im,carFound
