import numpy as np
import csv


def sigmoid(z):
    return np.clip(1/(1 + np.exp(-z)), 0.00000000000001, 0.99999999999999)

'''logistic regression'''

text = open('X_train','r')

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
'''

text = open('Y_train', 'r')
text.readline()

y = []
for r in text:
    r = [float(r)]
    y.append(r)

text.close()

X = np.array(x)
Y = np.array(y)

'''
X[:,0:3] = (X[:,0:3] - X[:,0:3].min(axis = 0)) / (X[:,0:3].max(axis = 0) - X[:,0:3].min(axis = 0))
X[:,4:6] = (X[:,4:6] - X[:,4:6].min(axis = 0)) / (X[:,4:6].max(axis = 0) - X[:, 4:6].min(axis = 0))
'''
'''
X[:, 1:] = (X[:, 1:] - X[:, 1:].min(axis = 0)) / (X[:, 1:].max(axis = 0) - X[:, 1:].min(axis = 0))
'''
X = (X - X.min(axis=0)) / (X.max(axis = 0) - X.min(axis = 0))


#add bias
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)


'''initial parameter '''

w = np.ones((X.shape[1], 1))

iteration = 14000
lr = 0.0001
s_grad = np.zeros((w.shape[0], 1))
lda = 0.1

loss = 0
for i in range(iteration):
    h = sigmoid(np.dot(X, w))
    loss = -(np.dot(Y.transpose(), np.log(h)) + np.dot((1 - Y).transpose(), np.log(1 - h)))
    grad = np.dot(X.transpose(), (h - Y))
    #grad[64:106] += 2 * lda * w[64:106]
    s_grad += grad**2
    ada = np.sqrt(s_grad)
    w = w - lr * grad
    

modelname = 'model_noada_goodfeature'
np.save('./model/' + modelname, w) 

h = sigmoid(np.dot(X, w))
for i in range(len(h)):
    if h[i] > 0.5:
        h[i] = 1
    else:
        h[i] = 0

accuracy = sum(np.logical_not(np.logical_xor(Y, h))) / X.shape[0]
print("done! loss: " + str(loss) + "accuracy: " + str(accuracy))

record = open('record', 'a+')
record.write('-----------------------------------------' + '\n')
record.write('iteration: ' + str(iteration) + 'lr: ' + str(lr) + '\n')
record.write('loss: ' + str(loss) + 'accuracy: ' + str(accuracy) + '\n')
record.write('model name: ' + modelname + '\n')
record.write('weight: ')
record.write(np.array_str(np.transpose(w)) + '\n')
record.close()





























