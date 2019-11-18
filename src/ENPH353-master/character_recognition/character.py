
import numpy as np
import cv2
import os
import characterModel

path = 'enph353_cnn_lab/pictures/'
files = os.listdir(path)
files_txt = [i for i in files if i.endswith('.png')]

# Load an color image in grayscale
img = cv2.imread(path + files_txt[25])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)

(thresh, bnw) = cv2.threshold(
    blur_gray, 127, 255, cv2.THRESH_BINARY)

# Taking a matrix of size 5 as the kernel
blackAndWhiteImage = bnw.copy()
kernel = np.ones((5, 5), np.uint8)
blackAndWhiteImage = cv2.dilate(blackAndWhiteImage, kernel, iterations=2)
blackAndWhiteImage = cv2.erode(blackAndWhiteImage, kernel, iterations=2)
blackAndWhiteImage = cv2.GaussianBlur(blackAndWhiteImage, (5, 5), 0)

# binary
ret, thresh = cv2.threshold(blackAndWhiteImage, 127,
                            255, cv2.THRESH_BINARY_INV)

# cv2.imshow('second', thresh)
# cv2.waitKey(0)

cv2MajorVersion = cv2.__version__.split(".")[0]
# check for contours on thresh
if int(cv2MajorVersion) == 4:
    ctrs, hier = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
else:
    im2, ctrs, hier = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
    roi = bnw[y:y + h, x:x + w]

# sort by max area
ROI = -1 * ROI
ROI = -1 * ROI[ROI[:, 0].argsort()]

# remove non characters
ROI = ROI[0:4, :]
# sort by x values
ROI = ROI[ROI[:, 1].argsort()]

model = characterModel.CharacterModel()
model.loadWeights()

augimg = img.copy()


for A, x, y, w, h in ROI:
    image = img[y:y + h, x:x + w]
    print(model.predict(image))

# print ROI
cv2.imshow('marked areas', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
