import csv
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

'''generative model'''

text = open('X_train', 'r')
text.readline()

x = []
for r in text:
    r = r.split('\n')
    r = r[0].split(',')
    for i in range(len(r)):
        r[i] = float(r[i])
    x.append(r)

text.close()

'''
normalize
use
(x - min) / (max - min)
'''

mx = x[0][0:6]
mn = x[0][0:6]
for i in range(len(x)):
    for j in range(6):
        if(x[i][j] > mx[j]):
            mx[j] = x[i][j]
        if(x[i][j] < mx[j]):
            mn[j] = x[i][j]

for i in range(len(x)):
    for j in range(6):
        x[i][j] = (x[i][j] - mn[j]) / (mx[j] - mn[j])

text = open('Y_train', 'r')
text.readline()

y = []
for r in text:
    r = [float(r)]
    y.append(r)

text.close()

X = np.array(x)
Y = np.array(y)

#add bias
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)



























