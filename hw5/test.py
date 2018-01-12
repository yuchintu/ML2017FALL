import sys
import numpy as np
import pandas as pd
from keras.models import load_model


def normalize(input):
    norm = np.linalg.norm(input, ord=1)
    if norm==0:
        norm=np.finfo(input.dtype).eps
    return input/norm

def load_data(fileName):
    dataFrame = pd.read_csv(fileName)
    user_id = dataFrame['UserID']
    user_id = user_id.apply(pd.to_numeric).as_matrix()-1
    movie_id = dataFrame['MovieID']
    movie_id = movie_id.apply(pd.to_numeric).as_matrix()-1
    return user_id, movie_id

'''
infile = open(sys.argv[1])
infile.readline()
for line in infile:
    temp = line.split("\n")
    temp = temp[0].split(",")
    user.append(int(temp[1]))
    movie.append(int(temp[2]))
'''


user, movie = load_data(sys.argv[1])

model = load_model('model_best.h5')
print(model.summary())
#movie_emb = np.array(model.layers[3].get_weights()).squeeze()
#np.save('movie_emb.npy', movie_emb)
result = model.predict([user,movie])
'''
#print(result)
out = open(sys.argv[2], 'w')
out.write('TestDataID,Rating\n')

for i in range(len(result)):
    out.write(str(i+1) + ',' + str(result[i][0]) + '\n')
'''





































