import csv
import numpy as np
import sys

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

'''
def PxC(x, u, sigma):
    return (1/np.power(2*np.pi, (sigma.shape[0]/2))) * (1/np.sqrt(np.linalg.det(sigma))) * np.exp(-1/2 * np.matmul(np.matmul((x - u), np.linalg.pinv(sigma)), np.transpose(x - u)))
'''

'''generative model'''

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
use
(x - min) / (max - min)
'''

mx = x[0][0:6]
mn = x[0][0:6]
for i in range(len(x)):
    for j in range(6):
        if(x[i][j] > mx[j]):
            mx[j] = x[i][j]
        if(x[i][j] < mn[j]):
            mn[j] = x[i][j]

for i in range(len(x)):
    for j in range(6):
        x[i][j] = (x[i][j] - mn[j]) / (mx[j] - mn[j])

text = open(sys.argv[2], 'r')
text.readline()

y = []
for r in text:
    r = [float(r)]
    y.append(r)

text.close()

X = np.array(x)
Y = np.array(y)

c1 = []  #higher than 50
c2 = []  #lower than 50

for i in range(X.shape[0]):
    if Y[i] == 1:
        c1.append(X[i])
    else:
        c2.append(X[i])

C1 = np.array(c1)
C2 = np.array(c2)
u1 = np.mean(C1, axis=0)
u2 = np.mean(C2, axis=0)
#print(C2)
sigma1 = np.dot(np.transpose(C1 - u1), (C1 - u1)) / len(C1)
sigma2 = np.dot(np.transpose(C2 - u2), (C2 - u2)) / len(C2)

sigma = (len(C1) / (len(C1) + len(C2))) * sigma1 + (len(C2) / (len(C1) + len(C2))) * sigma2
#print("1: " + str(sigma1) + "2: " + str(sigma2) + "3: " + str(sigma))
text = open(sys.argv[3], 'r')
text.readline()

x = []
for r in text:
    r = r.split('\n')
    r = r[0].split(',')
    for i in range(len(r)):
        r[i] = float(r[i])
    x.append(r)

text.close()

mx = x[0][0:6]
mn = x[0][0:6]
for i in range(len(x)):
    for j in range(6):
        if(x[i][j] > mx[j]):
            mx[j] = x[i][j]
        if(x[i][j] < mn[j]):
            mn[j] = x[i][j]

for i in range(len(x)):
    for j in range(6):
        x[i][j] = (x[i][j] - mn[j]) / (mx[j] - mn[j])


X = np.array(x)

'''
PCx = []
for i in range(len(X)):
    PCx.append((PxC(X[i], u1, sigma) * len(C1) / (len(C1) + len(C2))) / (PxC(X[i], u1, sigma) * len(C1) / (len(C1) + len(C2)) + PxC(X[i], u2, sigma) * len(C2) / (len(C1) + len(C2))))
'''

#print(u1.shape[0])


w = np.matmul((u1 - u2), np.linalg.pinv(sigma))
b = (-0.5)*np.dot(np.dot(u1, np.linalg.pinv(sigma)),np.transpose(u1)) + (0.5)*np.dot(np.dot(u2,np.linalg.pinv(sigma)),np.transpose(u2)) + np.log(len(C1)/len(C2))

ans = []
for i in range(len(X)):
    ans.append([str(i+1)])
    h = sigmoid(np.matmul(w, np.transpose(X[i])) + b)
    #print(h)
    if h > 0.5:
        ans[i].append(1)
    else:
        ans[i].append(0)

text = open(sys.argv[4], 'w')
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(['id', 'label'])
for i in range(len(ans)):
    s.writerow(ans[i])

text.close()



































