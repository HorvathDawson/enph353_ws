
import numpy as np
from character_recognition.character import find_license
import cv2
import os

path = 'licensePlateImages/'
files = os.listdir(path)
files_txt = [i for i in files if i.endswith('.png')]

# Load an color image in grayscale
# img = cv2.imread(path + files_txt[13])
# work on perspective transform num 14
img = cv2.imread(path + files_txt[13])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([90, 0, 70])
upper = np.array([130, 30, 200])
kernel = np.ones((20, 20), np.uint8)
mask = cv2.inRange(hsv, lower, upper)
kernel = np.ones((5, 5), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=2)
mask = cv2.erode(mask, kernel, iterations=2)

cv2.imshow("processed", mask)
cv2.waitKey(2000)
cv2.destroyAllWindows()

cv2MajorVersion = cv2.__version__.split(".")[0]
# check for contours on thresh
if int(cv2MajorVersion) == 4:
    ctrs, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
else:
    im2, ctrs, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

x_max, y_max, w_max, h_max = 0, 0, 0, 0
for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)
    if w_max * h_max < w * h:
        x_max, y_max, w_max, h_max = x, y, w, h
# cv2.rectangle(img,(x_max,y_max),(x_max+w_max,y_max+h_max),155,5)


# cv2.imshow("processed", img[y_max:y_max+h_max, x_max:x_max+w_max])
find_license(img[y_max:y_max+h_max, x_max:x_max+w_max].copy())
cv2.imshow("processed", img)
cv2.waitKey(5000)
cv2.destroyAllWindows()
