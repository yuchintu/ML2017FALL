import numpy as np 
import pickle

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from numpy import linalg as LA
from copy import deepcopy
from scipy import spatial

def read_data(filename='image.npy'):
    infile = np.load(filename)
    print(infile.shape)
    return infile

def draw(vector, label):
    print('draw')
    vector = np.array(vector, dtype=np.float64)
    vis_data = TSNE(n_components=2).fit_transform(vector)
    vis_x = vis_data[:,0]
    vis_y = vis_data[:,1]

    cm = plt.cm.get_cmap('RdYlBu')
    s = [0.5]
    plt.scatter(vis_x, vis_y, c=label, s=s)
    plt.show()

def PCA(array):
    S = array - np.mean(array, axis=0)
    S = S.transpose().dot(S)

    w, v = np.linalg.eig(S)
    
    d = dict()
    for i in range(len(w)):
        d[w[i]] = v[i]

    w = sorted(w, reverse=True)

    eigenvector = []
    for i in range(10):
        M = d[w[i]]
        M = M.reshape(784,1)
        eigenvector.append(M)
    
    eigenvector = np.array(eigenvector)
    eigenvector = eigenvector.reshape(eigenvector.shape[0], eigenvector.shape[1])
    return np.array(eigenvector)

def dimension_reduction(array, eigenvector):
    average = np.mean(array, axis=1).reshape(len(array),1)
    #print(average.shape)
    #print(eigenvector.shape)
    ratio = np.linalg.solve(eigenvector.dot(eigenvector.transpose()), eigenvector.dot((array-average).transpose()))
    ratio -= np.min(ratio)
    ratio = np.around(ratio/sum(ratio), 5)

    return ratio.transpose()

def dist(a, b): 
    t = []
    for i in range(len(b)):
        t.append(1 - spatial.distance.cosine(a[i], b[i]))
    return t

def k_mean(reduc_vector, threshold):
    k = 2
    C = np.random.rand(2, len(reduc_vector[0]))
    
    C_old = np.random.rand(2, len(reduc_vector[0]))

    clusters = np.zeros(len(reduc_vector))
    
    error = dist(C, C_old)
    #print(error)
    q = 0
    while q < threshold:
        for i in range(len(reduc_vector)):
            t = [reduc_vector[i], reduc_vector[i]]
            distances = dist(t, C)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        C_old = deepcopy(C)
        for i in range(k):
            points = [reduc_vector[j] for j in range(len(reduc_vector)) if clusters[j] == i]
            C[i] = np.mean(points, axis=0)
        print(C)
        error = dist(C, C_old)
        print(error)
        q += 1
        print(q)
    return clusters

def main():
    array = read_data()
    #eigenvector = PCA(array)
    #dim_reduc_vec = dimension_reduction(array, eigenvector)    
    array = np.array(array, dtype=np.float64)
    vis_data = TSNE(n_components=2).fit_transform(array)
    threshold = 10
    clusters = k_mean(dim_reduc_vec, threshold)
    print(clusters.shape)
   
    pickle.dump(clusters, open('clusters_'+str(threshold)+'.pkl', 'wb'))    
    draw(dim_reduc_vec[0:10000],clusters[0:10000])
    

if __name__ == '__main__':
    main()









