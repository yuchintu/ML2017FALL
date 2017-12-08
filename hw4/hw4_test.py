import numpy as np
from keras.models import load_model
from keras.preprocessing import sequence
import sys
import pickle
import sys

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

f = open(sys.argv[1])
f.readline()

x = []
for line in f:
    temp = line.split('\n')
    temp = temp[0].split(',')
    tmp = []
    for i in range(len(temp) - 1):
        tmp.append(temp[i+1])
    temp = ''
    for i in range(len(tmp)):
        temp += tmp[i]
    x.append(temp)

tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

#idx_dict = pickle.load(open('index_dict.pkl','rb'))
x = tokenizer.texts_to_sequences(x)
x = sequence.pad_sequences(x, maxlen = 50)
'''
t = ['today is a good day, but it is hot', 'today is hot, but it is a good day']
t = tokenizer.texts_to_sequences(t)
t = sequence.pad_sequences(t, maxlen = 40)
'''
model = load_model('model_best.h5')
result = model.predict(x)

out = open(sys.argv[2], 'w')
out.write('id,label\n')

for i in range(len(x)):
    if(result[i] > 0.5):
        out.write(str(i)+','+str(1)+'\n')
    else:
        out.write(str(i)+','+str(0)+'\n')

