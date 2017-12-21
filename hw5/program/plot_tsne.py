import numpy as np
import pandas as pd

def draw(x,y):
    from matplotlib import pyplot as plt
    from sklearn.manifold import TSNE
    y = np.array(y)
    x = np.array(x, dtype = np.float64)

    # perform t-SNE embedding
    vis_data = TSNE(n_components = 2).fit_transform(x)
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    cm = plt.cm.get_cmap('RdYlBu')
    sc = plt.scatter(vis_x, vis_y, c=y, cmap = cm)
    plt.colorbar(sc)
    plt.show()

def load_x(filename):
    x = np.load(filename)
    return x

def load_y(filename):
    y = pd.read_csv(filename)['Genres'].as_matrix().reshape(-1,1)
    return y

if __name__ == "__main__":
    x = load_x("movie_emb.npy")
    y = load_y("tsne_label.csv")
    print(x.shape)
    print(y.shape)
    draw(x, y)
