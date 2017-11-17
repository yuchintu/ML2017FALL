import keras
import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape
from keras.optimizers import SGD, Adam
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

model = load_model('model2.h5')
history = model.fit(X, Y, epochs = 100, batch_size = 128, validation_split = 0.1)

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
