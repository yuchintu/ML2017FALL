from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

def draw(x,y):
    y = np.array(y)
    x = np.array(x, dtype=np.float64)
    vis_data = TSNE(n_components=2).fit_transform(x)
    vis_x = vis_data[:,0]
    vis_y = vis_data[:,1]

    cm = plt.cm.get_cmap('RdYlBu')
    sc = plt.scatter(vis_x, vis_y, c=y, cmap=cm)
    plt.colorbar(sc)
    plt.show()

'''
infile = open('movies.csv')
infile.readline()

ID = []
Genres = []
for line in infile:
    line = line.split('\n')
    temp = line[0].split(',')
    t = ''
    for i in temp:
        t += i
    line = t.split('::')
    g = line[2].split('|')
    ID.append(line[0])
    Genres.append(g)
'''
movie_emb = np.load('movie_emb.npy')
y = pd.read_csv('tsne_label.csv')['Genres'].as_matrix().reshape(-1,1)
draw(movie_emb, y)
    




















