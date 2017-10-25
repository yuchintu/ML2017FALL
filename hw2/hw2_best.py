import csv
import numpy as np
import sys

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

text = open(sys.argv[1], 'r')

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
'''

w = np.load('./model/' + 'model_noada_goodre.npy')

X = np.array(x)

X[:,0:6] = (X[:,0:6] - X[:,0:6].min(axis=0)) / (X[:,0:6].max(axis=0) - X[:,0:6].min(axis=0))

X = np.concatenate((X, X**2), axis = 1)
X = np.concatenate((np.ones((X.shape[0],1)), X), axis = 1)

ans = []

for i in range(len(X)):
    ans.append([str(i+1)])
    h = sigmoid(np.dot(X[i], w))
    if(h[0] > 0.5):
        h[0] = 1
    else:
        h[0] = 0
    ans[i].append(int(h[0]))


filename = sys.argv[2]
text = open(filename, 'w+')
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(['id','label'])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()














































