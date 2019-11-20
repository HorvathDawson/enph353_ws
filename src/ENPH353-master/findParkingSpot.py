import numpy as np
import cv2
import os
from character_recognition.prespectiveTransform import extract_rect

path = 'licensePlateImages/'
files = os.listdir(path)
files_txt = [i for i in files if i.endswith('.png')]

def findParkingSpot():
	return -1


for i in range(len(files_txt)):
	img = cv2.imread(path + files_txt[i])

	kernel = np.ones((7,5), np.uint8) 

	img_erosion = cv2.erode(img, kernel, iterations=1) 

	mask = cv2.inRange(img_erosion,0,40)[40:,:-50]
	mask_dilate = cv2.dilate(mask,kernel,iterations=3)
	cv2.imshow("thres", img)
	cv2.imshow("mask",mask_dilate)
	cv2.waitKey(0)