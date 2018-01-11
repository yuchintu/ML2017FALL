import sys
import numpy as np
import pandas as pd

from keras.layers import Input, Dense
from keras.models import Model

from keras.models import load_model

from sklearn.cluster import KMeans

#builld model
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

# build encoder
encoder = Model(input=input_img, output=encoded)


#load model
encoder.load_weights('model_best.h5', by_name=True)

#load images
X = np.load(sys.argv[1])
X = X.astype('float32') / 255.
X = np.reshape(X, (len(X), -1))

#use encoder to encode image, and feed it into Kmeans
encoded_imgs = encoder.predict(X)
encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)
kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)


#get test case
f = pd.read_csv(sys.argv[2])
IDs, idx1, idx2 = np.array(f['ID']), np.array(f['image1_index']), np.array(f['image2_index'])


#predict
o = open(sys.argv[3], 'w')
o.write("ID,Ans\n")
for idx, i1, i2 in zip(IDs, idx1, idx2):
    p1 = kmeans.labels_[i1]
    p2 = kmeans.labels_[i2]
    if p1 == p2:
        pred = 1
    else:
        pred = 0
    o.write("{},{}\n".format(idx, pred))
o.close()












