import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import random as random

"""
training model - all data 
"""

csvfile = open('train.csv', encoding='Big5')

x = [] #record x

for row in csvfile:
    r = row.split('\n')
    d = r[0].split(',')
    x.append(d)
    
x.pop(0) #skip first row

#extract all data
f = [] #for X
y = [] #for Y
for row in x:
    row.pop(0)
    row.pop(0)
    if(row[0] == 'PM2.5'):
        y.append(row[1:(len(row) - 1)])
    if(row [0] == 'AMB_TEMP' or row[0] == 'CH4' or row[0] == 'NMHC' or row[0]== 'RAINFALL' or row[0] == 'THC' or row[0] == 'RH' or row[0] == 'WD_HR' or row[0] == 'WIND_DIREC' or row[0] == 'WIND_SPEED' or row[0] == 'WS_HR' or row[0] == 'CO' or row[0] == 'NO' or row[0] == 'NO2' or row[0] == 'NOx' or row[0] == 'O3' or row[0] == 'PM10' or row[0] == 'PM2.5' or row[0] == 'SO2'):
        row.pop(0)
        f.append(row)

for row in range(len(f)):
    for col in range(len(f[row])):
        if(f[row][col] == 'NR'):
            f[row][col] = 0.0
        f[row][col] = float(f[row][col])

datatype = int(18)
datalen = len(f[0])
X = [] #input
for n in range(int(len(f) / datatype)):
    tformix = []
    for i in range(datatype):
        tformix += f[datatype*n + i]
    for i in range(0, datalen - 10):
        tforX = []
        for j in range(datatype):
            tforX += tformix[(i + j * datalen) : (i + j * datalen + 9)] 
        X.append(tforX)

Y = [] #ground truth
for n in range(len(y)):
    for i in range(0, datalen - 10):
        Y.append(y[n][i + 9])
print(len(Y))
#y' = b + w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6 + w7*x7 + w8*x8 + w9*x9
#y' = b + w[1:9] * x[1:9]'

b = 1.0 #initial b
w = np.ones(len(X[0])) #initial w (w1,w2...,w9)
lr = 100 #initial learning rate
iteration = 1000

b_lr = 0.0
w_lr = 0.0

#Store_initial values for plotting
b_history = [b]
w_history = [w]

#Iteration
for i in range(iteration):
    
    b_grad = 0.0
    w_grad = np.zeros(len(w), dtype=np.float)
    for n in range(len(X)):
        wX = np.dot(w, X[n])
        #print("wX: " + str(wX))
        b_grad = b_grad - 2.0*(float(Y[n]) - b - wX)*1.0
        for j in range(len(w_grad)):
            w_grad[j] = w_grad[j] - 2.0*(float(Y[n]) - b - wX)*X[n][j]
    
    #print("bg: " + str(b_grad))
    #print("wg: " + str(w_grad[0]))
    
    b_lr = b_lr + b_grad**2
    w_lr = w_lr + w_grad**2
 
    # Update parameters
    b = b - lr/np.sqrt(b_lr) * b_grad
    w = w - lr/np.sqrt(w_lr) * w_grad

    # Store parameters for plotting
    b_history.append(b)
    w_history.append(w)

    error = 0
    for n in range(len(X)):
        wX = np.dot(w, X[n])
        error += (float(Y[n]) - (b + wX))**2
    error = error / len(X)
    print("i: " + str(i) + " error: " + str(error))

outfile = open('weight.txt','w')
outfile.write(str(b) + '\n')
outfile.write(str(w))
outfile.close()

print("b: " + str(b))
print(w)























