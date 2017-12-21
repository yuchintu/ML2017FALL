import numpy as np
import sys
import matplotlib.pyplot as plt

from keras.layers import Flatten, Input, Embedding, Reshape, Add,  Concatenate, Dropout, Dense, Dot
from keras.models import Sequential
from sklearn.utils import shuffle
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.merge import add, dot, concatenate
from keras.callbacks import ModelCheckpoint
from sklearn.cross_validation import train_test_split
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam

LATENT_DIMENSION = 50

def normalize(input):
    norm = np.linalg.norm(input, ord=1)
    print(norm)
    if norm==0:
        norm=np.finfo(input.dtype).eps
    return input/5 

def readData(filename, normal=False):
    user = []
    movie = []
    rating = []

    infile = open(filename)
    infile.readline()
    for line in infile:
        temp = line.split("\n")
        temp = temp[0].split(",")
        user.append(int(temp[1]))
        movie.append(int(temp[2]))
        rating.append(float(temp[3]))

    user, movie, rating = shuffle(user, movie, rating)
    
    user = np.array(user)
    movie = np.array(movie)
    rating = np.array(rating).astype(float)
      
    if normal == True:
        #user = normalize(user)
        #movie = normalize(movie)
        rating = normalize(rating)
        print(rating)

    
    return user, movie, rating

def plot_history(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def buildMF(n_user, n_movie, latent_dimension, bias=True):
    #model=Sequential()
    user_input = Input(shape=(1,))
    user_embedding = Embedding(input_dim=n_user, output_dim=latent_dimension)(user_input)
    user_out = Dropout(0.5)(Flatten()(user_embedding))

    movie_input = Input(shape=(1,))
    movie_embedding = Embedding(input_dim=n_movie, output_dim=latent_dimension)(movie_input)
    movie_out = Dropout(0.5)(Flatten()(movie_embedding))

    #predicted_preference = dot(inputs=[user_out, movie_out], axes=1)
    #predicted_preference = Flatten()(predicted_preference)
    predicted_preference = Dot(axes=1)([user_out,movie_out])
    
    if bias == True:
        user_bias = Embedding(n_user, 1)(user_input)
        user_bias = (Flatten()(user_bias))
    
        movie_bias = Embedding(n_movie, 1)(movie_input)
        movie_bias = (Flatten()(movie_bias))
    
        predicted_preference = Add()([predicted_preference,user_bias,movie_bias])

    model = Model(inputs=[user_input, movie_input], outputs=predicted_preference)
    adam = Adam(lr=0.001)
    model.compile(loss='mse',optimizer=adam)
    return model

def buildNN(n_user, n_movie, latent_dimension):
    user_input = Input(shape=(1,))
    user_embedding = Embedding(input_dim=n_user, output_dim=latent_dimension)(user_input)
    user_out = Dropout(0.5)(Flatten()(user_embedding))

    movie_input = Input(shape=(1,))
    movie_embedding = Embedding(input_dim=n_movie, output_dim=latent_dimension)(movie_input)
    movie_out = Dropout(0.5)(Flatten()(movie_embedding))
    merge_out = Concatenate()([user_out, movie_out])
    #hidden = Dense(150, activation='relu')(merge_out)
    hidden = Dense(150, activation='relu')(merge_out)
    hidden = Dense(50, activation='relu')(hidden)
    output = Dense(1)(hidden)

    model = Model([user_input, movie_input], output)
    model.compile(loss='mse', optimizer='sgd')
    return model

def main():
    user, movie, rating = readData(sys.argv[1], normal=False)
    n_user = len(user)
    n_movie = len(movie)
    
    model = buildMF(n_user, n_movie, LATENT_DIMENSION, bias=True)
    plot_model(model, to_file='model_MF.png')
    #model = buildNN(n_user, n_movie, LATENT_DIMENSION)
    #plot_model(model, to_file='model_nn.png')

    checkpoint = ModelCheckpoint('my_model_nn.h5',monitor = 'val_loss',save_best_only = True)
    history = model.fit([user,movie], rating, batch_size=1024, epochs=3, callbacks=[checkpoint],shuffle=False,validation_split=.1)
    plot_history(history)
    #movie_emb = np.array(model.layers[3].get_weights()).squeeze()
    #np.save('movie_emb.npy', movie_emb)

if __name__ == '__main__':
    main()
