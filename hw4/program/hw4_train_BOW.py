import numpy as np
import re
import pickle

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import Adam
from sklearn.cross_validation import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

import pickle


'''
def text_to_array(bow, sentences):
    new_sentences = []
    for i in range(len(sentences)):
        t = []
        for k in range(len(bow)):
            t.append(0)
        for j in range(len(sentences[i])):
            try:
                t[bow[sentences[i][j]]] += 1
            except:
                t = t
        new_sentences.append(t)
                
    return new_sentences
'''

EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 40


f = open('training_label.txt')

x = []
wx = []
y = []
for line in f:
    temp = line.split('+++$+++')
    y.append(temp[0])
    #x.append(re.findall(r"[\w]+", temp[1]))
    #wx.append(re.findall(r"[\w]+", temp[1]))
    x.append(line)

f.close()

f = open('testing_data.txt')
f.readline()

for line in f:
    temp = line.split(',')
    temp = temp[1].split('\n')
    #wx.append(re.findall(r"\w[\w]+", temp[0]))

f.close()


'''
bow = dict()
for i in range(len(wx)):
    for line in wx[i]:
        bow[line] = bow.get(line, 0) + 1

threshold = 1

wordvector = []
for word in bow:
    if bow[word] > threshold:
        wordvector.append(word)

print(len(wordvector))
bow = dict()
for i in range(len(wordvector)):
    bow[wordvector[i]] = i

print(bow)
'''

tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))


x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1)

y_train = np.array(y_train)
y_val = np.array(y_val)
x_train = tokenizer.texts_to_matrix(x_train, mode='count')
x_val = tokenizer.texts_to_matrix(x_val, mode='count')


model = Sequential()
model.add(Dense(32, activation = 'relu', input_shape = (x_train.shape[1],)))
model.add(Dropout(0.8))
model.add(Dense(1, activation = 'sigmoid'))

checkpoint = ModelCheckpoint('model_bow.h5', monitor = 'val_loss', save_best_only = True)
adam = Adam(lr = 1e-3)
model.compile(loss='binary_crossentropy', optimizer= adam, metrics=['accuracy'])

plot_model(model, to_file='model_bow.png')

history = model.fit(x_train, y_train, batch_size = 128, epochs = 5, validation_data = (x_val, y_val), callbacks = [checkpoint])

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





















































