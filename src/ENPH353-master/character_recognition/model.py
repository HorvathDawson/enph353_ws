from skimage import transform
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
import cv2
import os


# dimensions of our images.
img_width, img_height = 64, 64

train_data_dir = 'enph353_cnn_lab/TrainingData2/train'
validation_data_dir = 'enph353_cnn_lab/TrainingData2/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(36))
model.add(Activation('sigmoid'))
model.load_weights('weights.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=[0.7, 0.9],
    shear_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    rescale=1. / 255
)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical")

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical")

model.fit_generator(
    train_generator,
    steps_per_epoch=150,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=100)

test_loss, test_acc = model.evaluate(validation_generator, verbose=2)
print(test_acc)


path = 'enph353_cnn_lab/TrainingData2/validation/9/'
files = os.listdir(path)
files_txt = [i for i in files if i.endswith('.png')]


def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (64, 64, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


image = load(path + files_txt[3])
print(model.predict(image))

predicted_class_indices = np.argmax(model.predict(image), axis=1)
labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

print(predictions)

model.save_weights('weights.h5')


# import matplotlib.pyplot as plt
# #generate samples and plot
# for i in range(9):
#     # define subplot
#     plt.subplot(330 + 1 + i)
#     # generate batch of images
#     p = train_generator.next()
#     plt.imshow(p[0][0][:,:,0])
# # show the figure
# plt.show()
