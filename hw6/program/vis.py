import numpy as np

from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from matplotlib import pyplot as plt

def draw(vector, label):
    vector = np.array(vector, dtype=np.float64)
    label = np.array(label)
    vis_data = TSNE(n_components=2).fit_transform(vector)
    vis_x = vis_data[:,0]
    vis_y = vis_data[:,1]

    cm = plt.cm.get_cmap('RdYlBu')
    sc = plt.scatter(vis_x, vis_y, c=label, cmap=cm)
    plt.colorbar(sc)
    plt.show()


#load visualization.npy
V = np.load('visualization.npy')
V = V.astype('float32') / 255.
V = np.reshape(V, (len(V), -1))

#builld model
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

# build encoder
encoder = Model(input=input_img, output=encoded)

#load model
encoder.load_weights('model_best.h5', by_name=True)

#load model and encode image
encoded_imgs = encoder.predict(V)
encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)
kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)


label = []
#predict visualization and draw
for i in range(len(V)):
    label.append(kmeans.labels_[i])

draw(encoded_imgs, label)

#draw visualization with true answer
true_label = []
for i in range(5000):
    true_label.append(0)
for i in range(5000):
    true_label.append(1)

draw(encoded_imgs, true_label)




















