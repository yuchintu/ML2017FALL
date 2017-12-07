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
from keras import regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import sys

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

EMBEDDING_DIM = 60
MAX_SEQUENCE_LENGTH = 40
MAX_NB_WORDS = 1000

f = open(sys.argv[1])

x = []
wx = []
y = []
for line in f:
    temp = line.split('+++$+++')
    y.append(temp[0])
    #x.append(re.findall(r"[\w]+", temp[1]))
    x.append(temp[1])
    wx.append(temp[1])

f.close()

'''
f = open('testing_data.txt')
f.readline()

for line in f:
    temp = line.split(',')
    temp = temp[1].split('\n')
    wx.append(temp[0])
   
f.close()
'''

print("start token")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(wx)
print("finish token")

pickle.dump(tokenizer, open('tokenizer2.pkl', 'wb'))

'''
model = Word2Vec(wx, size = EMBEDDING_DIM, min_count=5, window=5, iter = 20)

index_dict, word_vectors = create_dictionaries(model)
train_wordidx = open('index_dict_word.pkl', 'wb')
pickle.dump(index_dict, train_wordidx)
train_wordidx.close()
'''

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1)
#x_train = text_to_index_array(index_dict, x_train)
#x_val = text_to_index_array(index_dict, x_val)
x_train = tokenizer.texts_to_sequences(x_train)
x_val = tokenizer.texts_to_sequences(x_val)
y_train = np.array(y_train)
y_val = np.array(y_val)

x_train = sequence.pad_sequences(x_train, MAX_SEQUENCE_LENGTH)
x_val = sequence.pad_sequences(x_val, MAX_SEQUENCE_LENGTH)

'''
n_symbols = len(index_dict) + 1
embedding_weights = np.zeros((n_symbols, EMBEDDING_DIM))
for w, index in index_dict.items():
    embedding_weights[index, :] = word_vectors[w]
'''

model = Sequential()
#model.add(Embedding(output_dim = EMBEDDING_DIM, input_dim = MAX_NB_WORDS, trainable=True, weights = [embedding_weights]))
model.add(Embedding(output_dim = EMBEDDING_DIM, input_dim = MAX_NB_WORDS, trainable=True))
#model.add(Conv1D(128, 1, strides = 3, activation = 'relu'))
#model.add(MaxPooling1D(pool_size=1, padding='valid'))
#model.add(LSTM(units = 128, activation = 'sigmoid', recurrent_activation = 'hard_sigmoid', return_sequences = True))
model.add(LSTM(units = 64, activation = 'sigmoid', recurrent_activation = 'hard_sigmoid'))
model.add(Dropout(0.5))

#model.add(Dense(32, activation = 'relu'))
#model.add(Dropout(0.5))

model.add(Dense(1, activation = 'sigmoid'))

model.summary()

checkpoint = ModelCheckpoint('model_best2.h5', monitor = 'val_loss', save_best_only = True)
adam = Adam(lr = 1e-3)
model.compile(loss='binary_crossentropy', optimizer= adam, metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size = 128, epochs = 5, validation_data = (x_val, y_val), callbacks = [checkpoint])

plot_model(model, to_file='model_rnn.png')

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

















































