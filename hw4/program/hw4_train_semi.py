import numpy as np
import re
import pickle

from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import Adam

from sklearn.cross_validation import train_test_split

from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary


def create_dictionaries(model):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
    w2indx = {v:k+1 for k, v in gensim_dict.items()}
    w2vec = {word: model[word] for word in w2indx.keys()}
    return w2indx, w2vec

def text_to_index_array(idx_dict, sentences):
    new_sentences =[]
    for s in sentences:
        t = []
        for word in s:
            try:
                t.append(idx_dict[word])
            except:
                t.append(0)
        new_sentences.append(t)
    return np.array(new_sentences)

EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 40


print('Reading data...')

f = open('training_label.txt')

x = []
wx = []
y = []
for line in f:
    #temp = line.split('\n')
    temp = line.split('+++$+++')
    y.append(int(temp[0]))
    #x.append(temp[1].split(' '))
    x.append(re.findall(r"[\w]+", temp[1]))
    wx.append(re.findall(r"[\w]+", temp[1]))

f.close()

print('train_label finish')

f = open('testing_data.txt')
f.readline()

for line in f:
    temp = line.split(',')
    temp = temp[1].split('\n')
    wx.append(re.findall(r"[\w]+", temp[0]))

f.close()

print('testing_data finish')

f = open('training_nolabel.txt')

nx = []
for line in f:
    temp = line.split('\n')
    temp = re.findall(r"[\w]+", temp[1])
    nx.append(temp)
    wx.append(temp)

f.close()

print('train_nolabel finish')

model = Word2Vec(wx, size = EMBEDDING_DIM, min_count=20, window=2)

index_dict, word_vectors = create_dictionaries(model)
train_wordidx = open('index_dict.pkl', 'wb')
pickle.dump(index_dict, train_wordidx)
train_wordidx.close()

print('semi supervised')

nx_p = text_to_index_array(index_dict, nx)
nx_p = sequence.pad_sequences(nx_p, maxlen = MAX_SEQUENCE_LENGTH)
self_model = load_model('model_best.h5')
ny = self_model.predict(nx_p)

for i in range(len(ny)):
    if ny[i] > 0.5:
        ny[i] = int(1)
    else:
        ny[i] = int(0)

print('semi finished')

x = x + nx
y = y + ny

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1)
x_train = text_to_index_array(index_dict, x_train)
x_val = text_to_index_array(index_dict, x_val)
y_train = np.array(y_train)
y_val = np.array(y_val)

x_train = sequence.pad_sequences(x_train, MAX_SEQUENCE_LENGTH)
x_val = sequence.pad_sequences(x_val, MAX_SEQUENCE_LENGTH)

n_symbols = len(index_dict) + 1
embedding_weights = np.zeros((n_symbols, EMBEDDING_DIM))
for w, index in index_dict.items():
    embedding_weights[index, :] = word_vectors[w]

model = Sequential()
#model.add(Embedding(output_dim = vec_dim, input_dim = n_symbols, mask_zero=True, weights = [embedding_weights], input_length = 40, trainable=False))
model.add(Embedding(output_dim = EMBEDDING_DIM, input_dim = n_symbols, weights = [embedding_weights], input_length = MAX_SEQUENCE_LENGTH, trainable=True))
model.add(Conv1D(filters=20, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(units = 100, activation = 'sigmoid', recurrent_activation = 'hard_sigmoid'))
model.add(Dropout(0.4))
model.add(Dense(1, activation = 'sigmoid'))
#model.add(Activation('sigmoid'))

checkpoint = ModelCheckpoint('model_best.h5', monitor = 'val_loss', save_best_only = True)
adam = Adam(lr = 1e-4)
model.compile(loss='binary_crossentropy', optimizer= adam, metrics=['accuracy'])

model.fit(x_train, y_train, batch_size = 128, epochs = 5, validation_data = (x_val, y_val), callbacks = [checkpoint])























































