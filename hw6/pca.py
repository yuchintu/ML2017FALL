import numpy as np
import sys

from skimage import io
from numpy import linalg as LA


import numpy as np
import sklearn.datasets, sklearn.decomposition
from skimage import transform


def normalize(matrix):
    matrix = matrix.astype(np.float32)
    matrix -= np.min(matrix)
    matrix /= np.max(matrix)
    matrix = np.array((matrix * 255).astype(np.uint8))
    return matrix

def show(matrix):
    io.imshow(normalize(matrix).reshape(300,300,3))
    io.show()

def reconstruction(Image, average, eigenface):
    '''
    ratio = np.linalg.solve(eigenface.transpose().dot(eigenface), eigenface.transpose().dot(Image-average))
    ratio -= np.min(ratio)
    ratio = np.around(ratio/sum(ratio), 1)
    
    recimg = average.reshape(600*600*3,1)
    for j in range(len(ratio)):
        recimg[:,0] += (eigenface[:,j] * ratio[j]).astype(np.uint8)
    ''' 
    #print(eigenface.shape)    #270000 * 4
    #print(Image.shape)
    #show(Image)
    #show(recimg)
    
def main():
    image = []
    for i in range(415):
        s = sys.argv[1] + '/' + str(i) + '.jpg'
        img = io.imread(s)
        img = transform.resize(img, (300, 300, 3))
        image.append(np.array(img).flatten())

    image = np.array(image).T    #270000 * 415
    print(image)
    #image = image.reshape(415,600*600*3).transpose()
    #print(image.shape)
    
    #draw average
    average = np.mean(image, axis=1)    #270000 * 1
    #show(average)

    U, s, V = np.linalg.svd(image, full_matrices=False)
    s = np.sqrt(s)
    sum_e = np.sum(s)
    for nb in range(4):
        XD = 'w'+str(nb+1)+'='
        print(XD,s[nb] / sum_e) 
    eigenface = U[:,0:4]
    print(eigenface)    
    img = int(sys.argv[2].split('.')[0])
    projection = eigenface.T.dot(image[:, img] - average)
    recimg = eigenface.dot(projection) + average
    recimg = normalize(recimg).reshape(300, 300, 3)
    recimg = transform.resize(recimg, (600, 600, 3))
    io.imsave('reconstruction.jpg', recimg)

if __name__ == '__main__':
    main()
    
    '''
    S = (image - average)
    S = S.dot(S.transpose())

    w, v = np.linalg.eig(S)

    d = dict()
    for i in range(len(w)):
        d[w[i]] = v[i]

    w = sorted(w, reverse=True)

    M = d[w[0]]
    M = M.reshape(415,1)
    M = (image-average).transpose().dot(M)
    eigenface = M
    show(eigenface)

    #draw top four eigenfaces
    for i in range(1,4):
        M = d[w[i]]
        M = M.reshape(415,1)
        M = (image-average).transpose().dot(M)
        eigenface = np.concatenate((eigenface, M), axis=1)
        M = M.reshape(600,600,3)    
        show(M)
    '''
    
    

