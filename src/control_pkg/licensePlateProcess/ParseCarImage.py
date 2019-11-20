
import numpy as np
from character_recognition.ReadText import find_license
from character_recognition.ReadText import find_ParkingSpot
from character_recognition.prespectiveTransform import extract_rect
import cv2
from character_recognition import characterModel
import os
from collections import Counter

TEAM_ID = "8"
TEAM_PASS = "*******"

model = characterModel.CharacterModel()
model.loadWeights()

def ParseCarImage(path):
    files = os.listdir(path)
    files_txt = [i for i in files if i.endswith('.png')]
    strs = [[],[],[],[],[],[],[],[],[],[]]
    licensePlates = []
    strs[0] = [str(TEAM_ID + "," + TEAM_PASS + ",0,0000")]
    for i in range(len(files_txt)):
        img = cv2.imread(path + files_txt[i])
        try:
            imgLicense, imgWarp, lx, ux = extract_rect(img)
        except:
            continue

        (h, w, c) = np.shape(imgLicense)
        if(w * h > 1500 and not imgLicense[0, 0, 0] == 155):
            try:
                # str = str(TEAM_ID +"," + TEAM_PASS + find_ParkingSpot(imgWarp[50:lx[1], lx[0]:ux[0]], model) + "," + find_license(imgLicense, model))
                index = int(find_ParkingSpot(imgWarp[50:lx[1], lx[0]:ux[0]], model))
                license = find_license(imgLicense, model)
                strs[index].append(license)
            except:
                continue
    # print(strs)
    for i in range(len(strs)):
        strs[i] = [license for license, license_count in Counter(strs[i]).most_common(1)]
    for i in range(len(strs)):
        if len(strs[i]) > 0:
            plate = strs[i][0]
            licensePlates.append(TEAM_ID +"," + TEAM_PASS + "," + str(i) + "," + plate)
    return licensePlates
