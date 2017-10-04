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

weight = open('weight.txt', 'r')
b = float(weight.readline())
w = weight.read()
w = w.split('[')
w = w[1].split(']')
w = w[0].split()
for i in range(len(w)):
    w[i] = float(w[i])


data = OrderedDict()
data['id'] = 'value'
for row in x:
    if(row[1] == 'PM2.5'):
        y = b + np.dot(w, row[2:11])
        data[row[0]] = y


s = pd.Series(data)
#s.to_csv('result.csv')
s.to_csv(sys.argv[2])
















