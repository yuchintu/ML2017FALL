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

weight = open('weight1.txt', 'r')
b = float(weight.readline())
w = weight.read()
w = w.split('[')
w = w[1].split(']')
w = w[0].split()
for i in range(len(w)):
    w[i] = float(w[i])

f = [] 
for row in x:
    if(row [1] == 'AMB_TEMP' or row[1] == 'CH4' or row[1] == 'NMHC' or row[1]== 'RAINFALL' or row[1] == 'THC' or row[1] == 'RH' or row[1] == 'WD_HR' or row[1] == 'WIND_DIREC' or row[1] == 'WIND_SPEED' or row[1] == 'WS_HR' or row[1] == 'CO' or row[1] == 'NO' or row[1] == 'NO2' or row[1] == 'NOx' or row[1] == 'O3' or row[1] == 'PM10' or row[1] == 'PM2.5' or row[1] == 'SO2'):
        f.append(row)

datatype = int(18) 
X = []

print(f[0])
for i in range(int(len(f) / datatype)):
    t = []
    t.append(f[i * datatype][0])
    #print(t)
    #print("t1: " + str(len(t)))
    for j in range(datatype):
        t += f[j + i * datatype][2:11]
        #print("t2: " + str(len(t)))
    X.append(t)

print(X[0])
for n in range(len(X)):
     for i in range(1, len(X[n])):
         X[n][i] = float(X[n][i])


data = OrderedDict()
data['id'] = 'value'
for row in X:
    print(len(row))
    y = b + np.dot(w, row[1:(9 * datatype) + 1])
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
















