import numpy as np
from keras.models import load_model
from keras.preprocessing import sequence
import sys
import pickle

f = open('testing_data.txt')
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
x = tokenizer.texts_to_matrix(x)
#x = sequence.pad_sequences(x, maxlen = 40)

model = load_model('model_bow.h5')
t = ['today is a good day, but it is hot', 'today is hot, but it is a good day']
t = tokenizer.texts_to_matrix(t)
result = model.predict(t)
print(result)
'''
out = open('result_bow.csv', 'w')
out.write('id,label\n')

for i in range(len(x)):
    if(result[i] > 0.5):
        out.write(str(i)+','+str(1)+'\n')
    else:
        out.write(str(i)+','+str(0)+'\n')
'''
