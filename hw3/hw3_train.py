import keras
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape, BatchNormalization
from keras.optimizers import SGD, Adam
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import sys

f = open(sys.argv[1])

f.readline()

x = []
y = []

for line in f:
    line = line.split(',')
    y.append(line[0])
    t = line[1].split('\n')
    t = t[0].split(' ')
    x.append(t)

X = np.array(x).astype(float) / 256.0
Y = keras.utils.to_categorical(np.array(y), num_classes = 7)

X = X.reshape(X.shape[0], 48, 48, 1)
XV = X[int(len(X)/10*9):]
YV = Y[int(len(X)/10*9):]
X = X[0:int(len(X)/10*9)]
Y = Y[0:int(len(Y)/10*9)]

model = Sequential()

model.add(Conv2D(128, (2,2), activation = 'relu', padding = 'same', input_shape = (48, 48, 1)))
model.add(BatchNormalization())
model.add(Conv2D(128, (2,2), activation = 'relu', padding = 'same'))
#model.add(MaxPooling2D((2,2), strides = (2,2), padding = 'same'))

model.add(BatchNormalization())
model.add(Conv2D(128, (2,2), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (2,2), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (2,2), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D((2,2), strides = (2,2), padding = 'same'))

model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
#model.add(MaxPooling2D((3,3), strides = (3,3), padding = 'same'))

model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D((3,3), strides = (3,3), padding = 'same'))

model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), activation = 'relu', padding = 'same'))
#model.add(MaxPooling2D((3,3), strides = (3,3), padding = 'same'))

model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D((3,3), strides = (3,3), padding = 'same'))


model.add(Flatten())

model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation = 'softmax'))
model.summary()

sgd = SGD(lr = 1e-4, decay = 1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr = 1e-4)

checkpoint = keras.callbacks.ModelCheckpoint('model_aug_best.h5', monitor = 'val_loss', save_best_only = True)

model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

datagen = ImageDataGenerator(featurewise_center = False, featurewise_std_normalization = False, rotation_range = 10, width_shift_range=0.1, height_shift_range = 0.1, horizontal_flip = True, vertical_flip = False)

datagen.fit(X)

model.fit_generator(datagen.flow(X, Y, batch_size = 128), steps_per_epoch=len(X)/128, epochs=150, validation_data = (XV, YV), callbacks = [checkpoint])

#model.fit(X, Y, epochs = 30, batch_size = 128, validation_split = 0.1, callbacks = [checkpoint])

#model.save('model_aug.h5')
























