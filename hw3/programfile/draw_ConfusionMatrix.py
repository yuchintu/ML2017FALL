#!/usr/bin/env python
# -- coding: utf-8 --

from keras.models import load_model
from sklearn.metrics import confusion_matrix
from utils import *
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def read_dataset(data_path):
    data = open(data_path, 'rb')
    train_pixels = pickle.load(data)
    for i in range(len(train_pixels)):
        train_pixels[i] = np.fromstring(train_pixels[i], dtype=float, sep=' ').reshape((48, 48, 1))
    return np.asarray(train_pixels)
    
def get_labels(data_path):
    data = open(data_path, 'rb')
    train_labels = pickle.load(data)
    train = []
    for i in range(len(train_labels)):
        train.append(int(train_labels[i]))
    return np.asarray(train)

def main():
    model_path = 'model_aug_best_66.h5'
    emotion_classifier = load_model(model_path)
    np.set_printoptions(precision=2)
    dev_feats = read_dataset('test_with_ans_pixels.pkl')
    dev_feats /= 256.0
    predictions = emotion_classifier.predict(dev_feats)
    predictions = predictions.argmax(axis=-1)
    print(predictions)
    te_labels = get_labels('test_with_ans_labels.pkl')
    print(te_labels)
    conf_mat = confusion_matrix(te_labels, predictions)
    
    plt.figure()
    plot_confusion_matrix(conf_mat, classes=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"])
    plt.show()

if __name__=='__main__':
    main()






































