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


model = characterModel.CharacterModel()
model.loadWeights()

path = 'enph353_cnn_lab/TrainingData2/validation/5/'
files = os.listdir(path)
files_txt = [i for i in files if i.endswith('.png')]


# def load(filename):
#     np_image = Image.open(filename)
#     return np_image
#
#
# image = load(path + files_txt[3])
# def blur(img):
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#     return (cv2.blur(img, (10, 10)))
#
# train_datagen = ImageDataGenerator(
#     rotation_range=40,
#     zoom_range=[0.6, 0.9],
#     shear_range=0.2,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     brightness_range=[0.7, 1.3],
#     horizontal_flip=False,
#     zca_whitening=True,
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rescale=1. / 255,
#     preprocessing_function=blur
# )
# data = img_to_array(image)
# samples = expand_dims(data, 0)
# testImages = train_datagen.flow(samples, batch_size=1)
#
# for i in range(2):
#     im = np.squeeze(testImages.next())
#     print(model.predict_keras(im))
#     cv2.imshow('here', im)
#     cv2.waitKey(0)

model.randomImgs()
# model.trainModel()

# model.trainModel()
# model.trainModel()
# model.saveWeights()
