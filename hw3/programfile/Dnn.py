import keras
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape, BatchNormalization
from keras.optimizers import SGD, Adam
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

f = open('train.csv')

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

model = Sequential()

model.add(Dense(64, activation = 'relu', input_shape = (48, 48, 1)))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(200, activation = 'relu'))
model.add(Flatten())

model.add(Dense(7, activation = 'softmax'))
model.summary()

adam = Adam(lr = 1e-4)

model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

plot_model(model, to_file='modelDnn.png')

history = model.fit(X, Y, epochs = 5, batch_size = 128, validation_split = 0.1)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()





















