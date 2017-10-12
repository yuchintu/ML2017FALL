import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import random as random

"""
training model - only PM2.5 
"""

csvfile = open('train.csv', encoding='Big5')

x = [] #record x

for row in csvfile:
    r = row.split('\n')
    d = r[0].split(',')
    x.append(d)
    
x.pop(0) #skip first row

#extract PM2.5
f = []
for row in x:    
    if(row[2] == 'PM2.5'):
        row.pop(0)
        row.pop(0)
        row.pop(0)
        f.append(row)

for row in range(len(f)):
    for col in range(len(f[row])):
        f[row][col] = float(f[row][col])

X = [] #input
Y = [] #ground truth
days = 10 #all = len(f)
for n in range(5,15):
    for i in range(0, len(f[n]) - 10):
        X.append(f[n][i:i+9])
        Y.append(f[n][i + 9])


#y' = b + w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6 + w7*x7 + w8*x8 + w9*x9
#y' = b + w[1:9] * x[1:9]'

b = 2.5 #initial b
w = np.ones(len(X[0])) #initial w (w1,w2...,w9)
for i in range(len(w)):
    w[i] = 0.5
lr = 100 #initial learning rate
iteration = 50000

b_lr = 0.0
w_lr = 0.0

#Store_initial values for plotting
b_history = [b]
w_history = [w]

#regularization
l = 0.1

#Iteration
for i in range(iteration):
    
    b_grad = 0.0
    w_grad = np.zeros(len(w))
    for n in range(len(X)):
        wX = np.dot(w, X[n])
        #print("wX: " + str(wX))
        b_grad = b_grad - 2.0*(float(Y[n]) - b - wX)*1.0
        for j in range(len(w_grad)):
            w_grad[j] = w_grad[j] - 2.0*(float(Y[n]) - b - wX)*X[n][j] - (l * 2 * w[j])
    
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
        error += (Y[n] - (b + wX))**2 + l * np.dot(w, w)
    error = np.sqrt(error / len(X))
    if(i % 100 == 0):
        print("i: " + str(i) + "error: " + str(error))

outfile = open('weightone1.txt','w')
outfile.write(str(b) + '\n')
outfile.write(str(w))
outfile.close()

print("b: " + str(b))
print(w)























