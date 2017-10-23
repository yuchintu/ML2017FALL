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
    
    '''
    t = []
    t += r[0:2]
    t += r[3:6]
    t += r[15:31]
    t += r[53:59]
    '''
    '''
    if r[2] == 1:
        r[2] = 2
    else:
        r[2] = -2
    del r[102] #delete US
    '''    
    x.append(r)
text.close()
 
'''
normalize
'''
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
'''

'''
w = np.load('./model/' + 'model_noada_goodfeature.npy')
'''

w = [[-6.99010475e+00,   1.86726958e+00,   1.04403661e+00,   8.41504142e-01,    2.42731146e+01,   2.72098608e+00,   2.89198240e+00 ,  7.05115894e-01,    3.83963826e-02,   6.95395042e-01 ,  2.16649433e-01,   3.91840884e-01,   -2.65121254e-01,  -9.47315043e-02,  -9.39006309e-01,   2.61356681e-01,   -2.74649871e-01,  -1.64277499e-01,   2.27146119e-01,  -8.01423624e-01,   -4.98822150e-01,  -7.31359872e-01,  -4.80701364e-01,   1.05959533e+00,    1.08219098e+00,   1.65482025e+00 ,  2.68401701e+00 ,  5.40124604e-01,    2.00728055e+00,  -1.66922119e+00,   2.49311929e+00,   8.82056683e-01,   -7.44816736e-01,   1.82845447e+00  , 1.39334684e+00 , -7.50927798e-01,   -1.21251606e+00 , -8.77824094e-01,  -6.25821367e-01,   6.89974673e-01 ,   3.12338029e-01,   7.61635280e-01,   1.46529472e+00,  -3.01438560e-01 ,   1.37509396e-02,   4.09055070e-01,  -1.42149859e-01 , -1.48462477e+00,    1.20341350e+00 ,  1.24906904e+00 ,  9.65824979e-01,   1.33613285e+00,    5.74867634e-01,  -4.32482772e-02,  -5.18932038e-01 ,  7.44733331e-03,   -9.56003844e-01,  -1.20466510e+00,  -1.33751725e-01 ,  8.15800618e-01 ,  -9.28508824e-01,  -2.94888864e-01,  -5.69787678e-01,  -7.94189942e-01 ,  -4.02729442e-01,   2.09082364e+00,   1.33267824e+00,   3.23512741e-01 ,  -8.44631763e-01,   1.35522293e+00,  -5.17501448e-01,   7.26058996e-01 ,   4.70256350e-01,   1.33831751e+00,   1.61552263e+00,   1.44863829e+00 ,   1.54566311e-01,   7.65139318e-01,   8.66110838e-01 ,  9.79068703e-01  ,  6.83418517e-01,   8.48641373e-01,   9.19832641e-01,   6.32731527e-01,    1.02789316e+00,   1.48014305e+00,   1.82082235e+00,   1.00692151e+00,    1.39132852e+00,   4.78019773e-01 ,  4.14519684e-01,   2.29454633e-01,    1.55056619e-02 ,  2.68020252e-01,   1.40785797e+00,   1.01334170e+00 ,   9.39277036e-01,   6.90125436e-01,   1.00934706e+00,  -7.36531302e-02,    1.00437189e+00 ,  4.93015694e-01 ,  6.27504796e-01,   1.21256529e+00,   -5.87867558e-02,   1.59602953e+00,   8.27862817e-01]]

print(len(w))
X = np.array(x)

'''
X[:,0:2] = (X[:,0:2] - X[:,0:2].min(axis=0)) / (X[:,0:2].max(axis=0) - X[:,0:2].min(axis=0))
X[:,3:6] = (X[:,3:6] - X[:,3:6].min(axis=0)) / (X[:,3:6].max(axis=0) - X[:,3:6].min(axis=0))
'''
X[:,0:6] = (X[:,0:6] - X[:,0:6].min(axis=0)) / (X[:,0:6].max(axis=0) - X[:,0:6].min(axis=0))

X = np.concatenate((np.ones((X.shape[0],1)), X), axis = 1)

ans = []

for i in range(len(X)):
    ans.append([str(i+1)])
    h = sigmoid(np.dot(X[i], np.transpose(w)))
    #h = np.dot(X[i], w)
    if(h[0] > 0.5):
        h[0] = 1
    else:
        h[0] = 0
    ans[i].append(int(h[0]))


filename = './res/' + 'resnoadagrad.csv'
text = open(filename, 'w+')
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(['id','label'])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()














































