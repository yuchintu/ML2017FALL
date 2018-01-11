import numpy as np
import jieba
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
import re
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from adjustText import adjust_text
from matplotlib.font_manager import FontProperties

def read_data(filename='all_sents.txt'):
    infile = open(filename)
    jieba.set_dictionary('dict.txt.big.txt')
    text = []
    for line in infile:
        line = line.split('\n')
        line = re.findall(r"[\w]+", line[0])
        line = jieba.cut(line[0])
        outline = " ".join(line)
        outline = outline.split()
        #print(outline)
        text.append(outline)
    return text

def text_to_dict(text):
    d = dict()
    for line in text:
        words = line.split()
        for word in words:
            d[word] = d.get(word, 0) + 1
    return d

def draw(vector, words):
    vector = np.array(vector, dtype=np.float64).reshape(len(vector),150)
    vis_data = TSNE(n_components=2).fit_transform(vector)
    vis_x = vis_data[:,0]
    vis_y = vis_data[:,1]

    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc', size=10)
    cm = plt.cm.get_cmap('RdYlBu')
    plt.scatter(vis_x, vis_y, c='b', edgecolor=(1,1,1,0))
    texts = []
    for x, y, w in zip(vis_x, vis_y, words):
        texts.append(plt.text(x, y, w, size=10, fontproperties=font))
    plt.title(str(adjust_text(texts, arrowprops=dict(arrowstyle='->', color='k', lw=0.5))))
    plt.show()

def main():
    text = read_data()
    '''
    dictionary = text_to_dict(text)
    #print(dictionary)
    words = []
    for word, count in dictionary.items():
        if count >= 3000:
            words.append([word])
    print(len(words))
    '''
    model = Word2Vec(text, size = 150, min_count=3000, window=5, iter = 20)
    #print(model.wv.vocab.keys())
    #print(model)
    
    words = []
    vector = []
    for word in model.wv.vocab.keys():
        words.append(word)
        vector.append(model[word])
    #print(len(vector))
    draw(vector, words)
    
if __name__ == '__main__':
    main()




