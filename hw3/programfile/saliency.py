import os
import argparse
from keras.models import load_model
from termcolor import colored,cprint
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from keras.visualizations import visualize_saliency
from vis.visualization import visualize_saliency

test_data = pd.read_csv('test.csv', sep = ',', encoding = 'UTF-8')
test_data = test_data.as_matrix()
id_data = test_data[:,0]
test_num = len(id_data)
test = list()
for i in range(test_num):
    temp = test_data[i][1].split()
    test.append(temp)
private_pixels = np.array(test).astype(int).reshape(test_num,48,48,1)/255.0
base_dir = os.path.dirname(os.path.dirname(os.path.realpath('train.csv')))

img_dir = os.path.join(base_dir, 'image')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
cmap_dir = os.path.join(img_dir, 'cmap')
if not os.path.exists(cmap_dir):
    os.makedirs(cmap_dir)
partial_see_dir = os.path.join(img_dir,'partial_see')
if not os.path.exists(partial_see_dir):
    os.makedirs(partial_see_dir)
row_dir = os.path.join(img_dir,'data')
if not os.path.exists(row_dir):
    os.makedirs(row_dir)
model_dir = os.path.join(base_dir, 'model')



parser = argparse.ArgumentParser(prog='plot_saliency.py',
        description='ML-Assignment3 visualize attention heat map.')
parser.add_argument('--epoch', type=int, metavar='<#epoch>', default=1)
args = parser.parse_args()
model_name = '.h5' #% str(args.epoch)
model_path = os.path.join(model_dir, model_name)
emotion_classifier = load_model('my_modelbest_dza_report.h5')
print(colored("Loaded model from {}".format(model_name), 'yellow', attrs=['bold']))
input_img = emotion_classifier.input
img_ids = 10

emotion_classifier.summary()
for idx in range(img_ids):
    img = private_pixels[idx].reshape(1,48,48,1)
    print(img)
    val_proba = emotion_classifier.predict(img)
    pred = val_proba.argmax(axis=-1)
    target = K.mean(emotion_classifier.output[:, pred])
    grads = K.gradients(target, input_img)[0]
    fn = K.function([input_img, K.learning_phase()], [grads])
    heatmap = visualize_saliency(emotion_classifier, layer_idx =11 ,seed_input = img, filter_indices = [1],)
    see = img.reshape(48, 48)
    
    '''
    plt.figure()
    plt.imshow(see,cmap='gray')
    plt.colorbar()
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    test_dir = os.path.join(row_dir, 'test')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    fig.savefig(os.path.join(test_dir, '{}.png'.format(idx)), dpi=100)
    heatmap = np.array(fn([img,1]))
    print("done!")
    '''
    print(heatmap)
    thres = 0.5
    for i in range(48):
        for j in range(48):
            if heatmap[i,j,0] == 0 and heatmap[i,j,2] <= heatmap[i,j,1] and heatmap[i,j,1] <=thres:
                see[i,j] = np.mean(see)
            elif heatmap[i,j,1] == 0 and heatmap[i,j, 0] == 0:
                see[i,j] = np.mean(see)


    plt.figure()
    plt.imshow(heatmap,cmap = plt.cm.jet)
    plt.colorbar()
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    test_dir = os.path.join(cmap_dir, 'test')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    fig.savefig(os.path.join(test_dir, '{}.png'.format(idx)), dpi=100)

    plt.figure()
    plt.imshow(see,cmap='gray')
    plt.colorbar()
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    test_dir = os.path.join(partial_see_dir, 'test')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    fig.savefig(os.path.join(test_dir, '{}.png'.format(idx)), dpi=100)
    
