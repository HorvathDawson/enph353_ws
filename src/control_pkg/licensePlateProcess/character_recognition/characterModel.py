from skimage import transform
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from noise import noisy

import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import heapq


class CharacterModel:
    def __init__(self, weight_path):
        K.clear_session()
        self.homePath = os.path.dirname(os.path.realpath(__file__))
        self.weight_path = weight_path
        # dimensions of our images.
        self.img_width, self.img_height = 64, 64

        train_data_dir = self.homePath + '/enph353_cnn_lab/TrainingData2/train'
        validation_data_dir = self.homePath + '/enph353_cnn_lab/TrainingData2/validation'

        if K.image_data_format() == 'channels_first':
            self.input_shape = (3, self.img_width, self.img_height)
        else:
            self.input_shape = (self.img_width, self.img_height, 3)
        batch_size = 128
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
        # self.model.add(Activation('sigmoid'))
        self.model.add(Activation('softmax'))
        self.model.compile(
                           # optimizer='rmsprop',
                           # loss='binary_crossentropy',
                           loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        # this is the augmentation configuration we will use for training
        def random_colour(img):
            colour_mode = random.randint(1, 4)
            if colour_mode == 1:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif colour_mode == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            return img
        def random_pic(img):
            blur_value = random.randint(13, 18)
            if random.choice([0, 1]):
                # if random.choice([0, 1]):
                #     img = noisy('gauss',img)
                if random.choice([0, 1]):
                    img = noisy('s&p',img)
                if random.choice([0, 1]):
                    img = noisy('poisson',img)
                # if random.choice([0, 1]):
                #     img = noisy('speckle',img)
            if random.choice([0, 1]):
                img = cv2.blur(img, (blur_value, blur_value))
            else:
                img = cv2.blur(img, (5, blur_value))

            if random.choice([0, 1]):
                blr = random.randint(20,75)
                img = cv2.addWeighted(img, 4, cv2.blur(img, (blr, blr)), -3, 64)
                img = img.clip(min=0)
            img = img.astype(np.float32)
            scale_percent = 20 # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            dim_orig = (img.shape[1], img.shape[0])

            # resize image
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            img = cv2.resize(resized, dim_orig, interpolation = cv2.INTER_AREA)
            img = random_colour(img)

            return img


        train_datagen = ImageDataGenerator(
            rotation_range=25,
            zoom_range=[0.9, 1.1],
            shear_range=0.3,
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
        # cv2.imshow("..", np_image)
        # cv2.waitKey(0)
        np_image = transform.resize(np_image, self.input_shape)
        np_image = np.expand_dims(np_image, axis=0)
        predict = self.model.predict(np_image)
        k = np.array((predict.copy())[0], dtype=np.float128)
        largest_ind = np.argpartition(k, [0,-1,-2,-3,-4,-5])[-5:][::-1]

        predicted_class_indices = np.argmax(predict, axis=1)
        labels = (self.train_generator.class_indices)
        labels = dict((v, k) for k, v in labels.items())

        return [labels[k] for k in largest_ind]

    def loadWeights(self):
        self.model.load_weights(self.homePath + self.weight_path)

    def saveWeights(self):
        self.model.save_weights(self.homePath + self.weight_path)

    def trainModel(self):
        history_conv = self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=50,
            epochs=5,
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
