import sys
import pandas as pd
import numpy as np
from collections import OrderedDict

#input data
csvfile = open(sys.argv[1], encoding='Big5')

x = []

for row in csvfile:
    r = row.split('\n')
    d = r[0].split(',')
    x.append(d)

for i in range(len(x)):
    for j in range(2,len(x[i])):
        if(x[i][j] != "NR"):        
            x[i][j] = float(x[i][j])
        else:
            x[i][j] = 0.0
csvfile.close()

'''
weight = open('weight69.txt', 'r')
b = float(weight.readline())
w = weight.read()
w = w.split('[')
w = w[1].split(']')
w = w[0].split()
for i in range(len(w)):
    w[i] = float(w[i])
'''

b = 2.88057709857
w = [5.33802394e-02, 3.14086616e-02, 5.53665721e-02, -1.20635619e-03, -2.23076129e-02, 1.11337931e-01, -1.93567915e-01, 9.51990995e-02, 7.33451284e-01, -2.37763809e-04, -4.31160933e-03, 7.59334523e-03, -7.83792581e-03, 1.48636823e-03, 7.39340099e-03, -6.87113482e-03, -2.72932373e-03, 6.60554068e-03]


f = [] 
for row in x:
    if(row[1] == 'PM2.5'):
        f.append(row)

datatype = int(1) 
X = []

print(f[0])
for i in range(int(len(f) / datatype)):
    t = []
    t.append(f[i * datatype][0])
    #print(t)
    #print("t1: " + str(len(t)))
    for j in range(datatype):
        t += f[j + i * datatype][2:11]
        for k in range(2, 11):
            t.append(f[j + i * datatype][k]**2)
    #print("t2: " + str(len(t)))
    X.append(t)

#print(X[0])
for n in range(len(X)):
     for i in range(1, len(X[n])):
         X[n][i] = float(X[n][i])


data = OrderedDict()
data['id'] = 'value'
for row in X:
    #print(len(row))
    y = b + np.dot(w, row[1:9*2 + 1])
    data[row[0]] = y

'''
for row in x:
    if(row[1] == 'PM2.5'):
        y = b + np.dot(w, row[2:11])
        data[row[0]] = y
'''

s = pd.Series(data)
#s.to_csv('result.csv')
s.to_csv(sys.argv[2])
















