import numpy as np
import pandas as pd
'''
# read movieID in training data
train_id = pd.read_csv("train.csv")['MovieID'].as_matrix().reshape(-1,1)
print(train_id.shape)
'''
# get genres of each movieID in movies
movie_id = pd.read_csv("movies.csv", sep = '::', engine = 'python')['movieID'].tolist()
movie_genres = pd.read_csv("movies.csv", sep = '::', engine = 'python')['Genres'].str.split('|').str[-1].tolist()

# to see how many genres there are
'''
genres = {'genres'}
for idx in range(len(movie_genres)):
    if movie_genres[idx] not in genres:
        genres.add(movie_genres[idx])
print(genres)
'''

num_0 = 0
num_1 = 0
num_2 = 0
num_3 = 0
num_4 = 0
num_5 = 0
#3952
for idx in range(len(movie_genres)):
    if movie_genres[idx] in {'Drama', 'Musical'}:
            movie_genres[idx] = 0
            num_0 += 1
    elif movie_genres[idx] in {'Thriller', 'Horror', 'Mystery', 'Crime', 'Action', 'War'}:
            movie_genres[idx] = 1
            num_1 += 1
    elif movie_genres[idx] in {'Animation', 'Children\'s', 'Comedy'}:
            movie_genres[idx] = 2
            num_2 += 1
    elif movie_genres[idx] in {'Adventure', 'Fantasy', 'Sci-Fi'}:
            movie_genres[idx] = 3
            num_3 += 1
    elif movie_genres[idx] in {'Film-Noir', 'Documentary'}:
            movie_genres[idx] = 4
            num_4 += 1
    else:
            movie_genres[idx] = 5
            num_5 += 1
'''
print(num_0)
print(num_1)
print(num_2)
print(num_3)
print(num_0 + num_1 + num_2 + num_3)
'''
token_genres = []
idx = 0
for i in range(3952):
    if i in movie_id:
        token_genres.append(movie_genres[idx])
        idx += 1
    else:
        token_genres.append(5)
#print(len(token_genres))
token_genres = np.array(token_genres).reshape(-1,1)
#print(token_genres.shape)

'''
# build dictionary
movie_dict = np.hstack((movie_id, movie_genres))
#print(movie_dict.shape)

train_genres = []
for idx in range(len(train_id)):
    row_num = np.where(movie_dict[:,0] == train_id[idx])[-1]
    train_genres += movie_dict[row_num,1].tolist()

train_genres = np.array(train_genres).reshape(-1,1)
print(train_genres.shape)
'''
# generate prediction file
output = token_genres
output_df = pd.DataFrame(data = output, columns = ['Genres'])
output_df.to_csv("tsne_label.csv", index = False)
