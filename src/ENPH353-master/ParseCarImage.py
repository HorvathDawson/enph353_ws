
import numpy as np
from character_recognition.ReadText import find_license
from character_recognition.prespectiveTransform import extract_rect
import cv2
from character_recognition import characterModel
import os


path = 'licensePlateImages/'
files = os.listdir(path)
files_txt = [i for i in files if i.endswith('.png')]
TEAM_ID = "BITCHING"
TEAM_PASS = "*******"

model = characterModel.CharacterModel()
model.loadWeights()


for i in range(len(files_txt)):
    img = cv2.imread(path + files_txt[i])
    try:
        imgLicense, imgWarp, lx, ux = extract_rect(img)
    except:
        continue

    (h, w, c) = np.shape(imgLicense)
    if(w * h > 1500 and not imgLicense[0, 0, 0] == 155):
        try:
            print(find_license(imgLicense, model))
        except:
            cv2.imshow("parking Spot", imgWarp[50:lx[1], lx[0]:ux[0]])
            cv2.imshow("License", imgLicense)
            cv2.waitKey(0)
