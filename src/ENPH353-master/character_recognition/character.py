
import numpy as np
import cv2
import os
import characterModel

# path = 'enph353_cnn_lab/pictures/'
# files = os.listdir(path)
# files_txt = [i for i in files if i.endswith('.png')]
#
# # Load an color image in grayscale
# img = cv2.imread(path + files_txt[25])
def find_license(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([115, 110, 60])
    upper = np.array([125, 255, 255])
    kernel = np.ones((20, 20), np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((1, 1), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)

    cv2.imshow('second', mask)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

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

    model = characterModel.CharacterModel()
    model.loadWeights()

    augimg = img.copy()


    for A, x, y, w, h in ROI:
        image = mask[y-2:y + h + 2, x-2:x + w+2]
        image = cv2.bitwise_not(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # cv2.imshow('marked areas', image)
        # cv2.waitKey(2500)
        print(model.predict(image))

# print ROI
# cv2.imshow('marked areas', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
