import csv
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


'''
text = open('test.csv', 'r')
c = csv.reader(text, delimiter=',')
next(c)

workclass = {' Federal-gov':0, ' Local-gov':1,' Never-worked':2, ' Private':3, ' Self-emp-inc':4, ' Self-emp-not-inc': 5, 
' State-gov':6, ' Without-pay':7,' ?': 8}
education = {' 10th':0,' 11th':1,' 12th':2,' 1st-4th':3,' 5th-6th':4,' 7th-8th':5,' 9th':6,' Assoc-acdm':7,' Assoc-voc':8,' Bachelors':9,' Doctorate':10,' HS-grad':11,' Masters':12,' Preschool':13,' Prof-school':14,' Some-college':15}
marital = {' Divorced':0,' Married-AF-spouse':1,' Married-civ-spouse':2,' Married-spouse-absent':3,' Never-married':4,' Separated':5,' Widowed':6} 
occupation = {' Adm-clerical':0,' Armed-Forces':1,' Craft-repair':2,' Exec-managerial':3,' Farming-fishing':4,' Handlers-cleaners':5,' Machine-op-inspct':6,' Other-service':7,' Priv-house-serv':8,' Prof-specialty':9,' Protective-serv':10,' Sales':11,' Tech-support':12,' Transport-moving':13,' ?':14} 
relationship = {' Husband':0,' Not-in-family':1,' Other-relative':2,' Own-child':3,' Unmarried':4,' Wife':5} 
race = {' Amer-Indian-Eskimo':0,' Asian-Pac-Islander':1,' Black':2,' Other':3,' White':4} 
native_country = {' Cambodia':0,' Canada':1,' China':2,' Columbia':3,' Cuba':4,' Dominican-Republic':5,' Ecuador':6,' El-Salvador':7,' England':8,' France':9,' Germany':10,' Greece':11,' Guatemala':12,' Haiti':13,' Holand-Netherlands':14,' Honduras':15,' Hong':16,' Hungary':17,' India':18,' Iran':19,' Ireland':20,' Italy':21,' Jamaica':22,' Japan':23,' Laos':24,' Mexico':25,' Nicaragua':26,' Outlying-US(Guam-USVI-etc)':27,' Peru':28,' Philippines':29,' Poland':30,' Portugal':31,' Puerto-Rico':32,' Scotland':33,' South':34,' Taiwan':35,' Thailand':36,' Trinadad&Tobago':37,' United-States':38,' Vietnam':39,' Yugoslavia':40,' ?':41}

x = []
for r in c:
    t = []
    t.append(float(r[0]))  #age
    t.append(float(r[2]))  #fnlwgt
    if(r[9] == ' Male'):   #sex
        t.append(1.0)
    else:
        t.append(0.0)
    t.append(float(r[10])) #capital_gain
    t.append(float(r[11])) #capital_loss
    t.append(float(r[12])) #hours_per_week
    #workclass
    w = [0,0,0,0,0,0,0,0,0]
    w[workclass.get(r[1])] = 1
    t += w
    #education
    e = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    e[education.get(r[3])] = 1
    t += e
    #marital_status
    m = [0,0,0,0,0,0,0]
    m[marital.get(r[5])] = 1
    t += m
    #occupation
    o = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    o[occupation.get(r[6])] = 1
    t += o
    #relationship
    re = [0,0,0,0,0,0]
    re[relationship.get(r[7])] = 1
    t += re
    #race
    ra = [0,0,0,0,0]
    ra[race.get(r[8])] = 1
    t += ra
    #native_country
    na = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    na[native_country.get(r[13])] = 1
    t += na
    
    x.append(t)
text.close()
'''

text = open('X_test', 'r')

text.readline()

x = []
for r in text:
    r = r.split('\n')
    r = r[0].split(',')
    for i in range(len(r)):
        r[i] = float(r[i])
    del r[102] #delete US
    x.append(r)
text.close()
 
'''
normalize
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


w = np.load('model_noada_delus.npy')
print(w)
X = np.array(x)
'''
X[:,0:6] = (X[:,0:6] - X[:,0:6].min(axis=0)) / (X[:,0:6].max(axis=0) - X[:,0:6].min(axis=0))
'''
#print(X[0])
#print(X[1])

X = np.concatenate((np.ones((X.shape[0],1)), X), axis = 1)

ans = []

for i in range(len(X)):
    ans.append([str(i+1)])
    h = sigmoid(np.dot(X[i], w))
    #h = np.dot(X[i], w)
    if(h[0] > 0.5):
        h[0] = 1
    else:
        h[0] = 0
    ans[i].append(int(h[0]))


filename = 'resnoadadelus.csv'
text = open(filename, 'w+')
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(['id','label'])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()














































