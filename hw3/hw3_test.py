from keras.models import load_model
import numpy as np
import csv
import sys

'''
def maxValue(ans):
    m = ans[0]
    index = 0
    for i in range(len(ans)):
        if ans[i] > m:
            index = i
    return index
'''

f = open(sys.argv[1])

f.readline()

x = []

for line in f:
    line = line.split(',')
    t = line[1].split('\n')
    t = t[0].split(' ')
    x.append(t)

X = np.array(x).astype(float) / 256.0

X = X.reshape(X.shape[0], 48, 48, 1)

model1 = load_model('model_aug_best_66.h5')    
ans1 = model1.predict(X)
ans1 = np.argmax(ans1, axis = -1)

model2 = load_model('model_aug_best_66731.h5')
ans2 = model2.predict(X)
ans2 = np.argmax(ans2, axis = -1)

model3 = load_model('model_aug_best_65.h5')
ans3 = model3.predict(X)
ans3 = np.argmax(ans3, axis = -1)

ans = []

for i in range(len(X)):
    d = dict()
    d[ans1[i]] = d.get(ans1[i], 0) + 1
    d[ans2[i]] = d.get(ans2[i], 0) + 1
    d[ans3[i]] = d.get(ans3[i], 0) + 1
    t = dict()
    for key, value in d.items():
        t[value] = key
    m = max(t)
    m = t[m]
    ans.append(m)

#print(len(ans))
#print(ans)

text = open(sys.argv[2], 'w+')
s = csv.writer(text, delimiter = ',', lineterminator = '\n')
s.writerow(['id','label'])
for i in range(len(ans)):
    s.writerow([i, ans[i]])
text.close()





















