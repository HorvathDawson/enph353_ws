from skimage import transform
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os


class CharacterModel:
    def __init__(self):
        # dimensions of our images.
        self.img_width, self.img_height = 64, 64

        train_data_dir = 'enph353_cnn_lab/TrainingData2/train'
        validation_data_dir = 'enph353_cnn_lab/TrainingData2/validation'

        if K.image_data_format() == 'channels_first':
            self.input_shape = (3, self.img_width, self.img_height)
        else:
            self.input_shape = (self.img_width, self.img_height, 3)
        batch_size = 32
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=self.input_shape))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(36))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])
        # this is the augmentation configuration we will use for training
        def random_colour(img):
            blur_value = random.randint(3, 8)
            colour_mode = random.randint(1, 5)
            if colour_mode == 1:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                img = cv2.blur(img, (blur_value, blur_value))
            if colour_mode == 2:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = cv2.blur(img, (blur_value, blur_value))
            if colour_mode == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.blur(img, (blur_value, blur_value))
            if colour_mode == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
                img = cv2.blur(img, (blur_value, blur_value))
            return img
        def random_pic(img):
            blur_value = random.randint(3, 8)
            if random.randint(1,4) == 1:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
                (thresh, blackAndWhiteImage) = cv2.threshold(
                    blur_gray, 127, 255, cv2.THRESH_BINARY)
                blackAndWhiteImage = cv2.cvtColor(blackAndWhiteImage, cv2.COLOR_GRAY2RGB)
                # Mask used to flood filling.
                # Notice the size needs to be 2 pixels than the image.
                h, w = blackAndWhiteImage.shape[:2]
                mask = np.zeros((h + 2, w + 2), np.uint8)

                # Floodfill
                corner = random.randint(1, 3)
                if random.randint(1, 3) == 1:
                    cv2.floodFill(img, mask, (0, 0), (random.randint(
                        1, 255), random.randint(1, 255), random.randint(1, 255)))
                if random.randint(1, 3) == 1:
                    cv2.floodFill(img, mask, (0, w - 2), (random.randint(
                        1, 255), random.randint(1, 255), random.randint(1, 255)))
                if random.randint(1, 3) == 1:
                    cv2.floodFill(img, mask, (h - 2, 0), (random.randint(
                        1, 255), random.randint(1, 255), random.randint(1, 255)))
                if random.randint(1, 3) == 1:
                    cv2.floodFill(img, mask, (h - 2, w - 2), (random.randint(
                        1, 255), random.randint(1, 255), random.randint(1, 255)))
                if random.randint(1,6) > 3:
                    img = cv2.blur(img, (blur_value, blur_value))
            img = random_colour(img)
            return img
        train_datagen = ImageDataGenerator(
            rotation_range=40,
            zoom_range=[0.6, 0.9],
            shear_range=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1,
            brightness_range=[0.7, 1.3],
            horizontal_flip=False,
            zca_whitening=True,
            featurewise_center=True,
            featurewise_std_normalization=True,
            rescale=1. / 255,
            preprocessing_function=random_pic
        )

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        self.train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=batch_size,
            class_mode="categorical")

        self.validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=batch_size,
            class_mode="categorical")

    def predict(self, np_image):
        np_image = transform.resize(np_image, self.input_shape)

        # plt.imshow(np_image)
        # plt.show()
        # cv2.imshow('image for predicition', np_image)
        # cv2.waitKey(0)

        np_image = np.expand_dims(np_image, axis=0)
        predicted_class_indices = np.argmax(
            self.model.predict(np_image), axis=1)
        labels = (self.train_generator.class_indices)
        labels = dict((v, k) for k, v in labels.items())
        return [labels[k] for k in predicted_class_indices]

    def loadWeights(self):
        self.model.load_weights('weights.h5')

    def saveWeights(self):
        self.model.save_weights('weights.h5')

    def trainModel(self):
        self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=1000,
            epochs=1,
            validation_data=self.validation_generator,
            validation_steps=100)

        test_loss, test_acc = self.model.evaluate(
            self.validation_generator, verbose=2)
        print(test_acc)

    def randomImgs(self):
        # generate samples and plot
        for i in range(9):
            # define subplot
            plt.subplot(330 + 1 + i)
            # generate batch of images
            p = self.train_generator.next()
            plt.imshow(p[0][0][:, :, :])
        # show the figure
        plt.show()
