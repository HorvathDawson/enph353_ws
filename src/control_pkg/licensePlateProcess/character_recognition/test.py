import characterModel
from skimage import transform
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os

from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator


model = characterModel.CharacterModel('/weights.h5')
model.loadWeights()


# model.randomImgs()
model.trainModel()


model.saveWeights()
