#!/usr/bin/env python
# -- coding: utf-8 --
import os
import sys
import argparse
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from termcolor import colored,cprint
import numpy as np
from utils import * 
import pickle

basedir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
exp_dir = 'exp'
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
vis_dir = os.path.join('image','vis_layer')
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)
filter_dir = os.path.join('image','vis_filter')
if not os.path.exists(filter_dir):
    os.makedirs(filter_dir)

nb_class = 7
LR_RATE = 1e-2
NUM_STEPS = 10
RECORD_FREQ = 10

def deprocess_image(x):
    """
    As same as that in problem 4.
    """
    """
    As same as that in problem 4.
    """
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    #x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')

    return x

def main():
    parser = argparse.ArgumentParser(prog='visFilter.py',
            description='Visualize CNN filter.')
    parser.add_argument('--model',type=str,default='simple',choices=['simple','NTUEE'],
            metavar='<model>')
    parser.add_argument('--epoch',type=int,metavar='<#epoch>',default=20)
    parser.add_argument('--mode',type=int,metavar='<visMode>',default=2,choices=[1,2])
    parser.add_argument('--batch',type=int,metavar='<batch_size>',default=64)
    parser.add_argument('--idx',type=int,metavar='<suffix>',default=17, required=False)
    args = parser.parse_args()
    #store_path = "{}_epoch{}{}".format(args.model,args.epoch,args.idx)
    store_path = 'model1.h5'
    print(colored("Loading model from {}".format(store_path),'yellow',attrs=['bold']))
    #modelpath = os.path.join(expdir,storepath,'model.h5')
    #emotion_classifier = load_model('model/model-{}.h5'.format(args.epoch))
    emotion_classifier = load_model(store_path)
    emotion_classifier.summary()


    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])

    def normalize(x):
        # utility function to normalize a tensor by its L2 norm
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

    def grad_ascent(num_step,input_image_data,iter_func):
        """
        Implement this function!
        """
        filter_images = input_image_data
        step = 1
        iterate = iter_func
        for i in range(num_step):
            loss_value, grads_value = iterate([input_image_data])
            input_image_data += grads_value * step

            #print('Current loss value:', loss_value)
            #if loss_value <= 0.:
                # some filters get stuck to 0, we can skip them
            #    break

        # decode the resulting input image
        #if loss_value > 0:
        filter_images = deprocess_image(input_image_data[0])
        #print(filter_images.shape)
        #print(filter_images)
        #print(loss_value)    
        return filter_images.reshape(48, 48), loss_value
    
    def load_pickle(data_name):
        f = open(data_name, 'rb')
        return pickle.load(f)

    input_img = emotion_classifier.input
    # visualize the area CNN see
    if args.mode == 1:
        collect_layers = list()
        collect_layers.append(K.function([input_img,K.learning_phase()],[layer_dict['conv2d_4'].output]))

        dev_feat = load_pickle('test_with_ans_pixels.pkl')
        dev_label = load_pickle('test_with_ans_labels.pkl')
        choose_id = 17
        photo = dev_feat[choose_id]
        photo = photo.split()
        for p in photo:
            p = int(p)
        photo = np.array(photo)
        for cnt, fn in enumerate(collect_layers):
            im = fn([photo.reshape(1,48,48,1),0])
            fig = plt.figure(figsize=(14,8))
            nb_filter = im[0].shape[3]
            for i in range(nb_filter):
                ax = fig.add_subplot(nb_filter/16,16,i+1)
                ax.imshow(im[0][0,:,:,i],cmap='Purples')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.tight_layout()
            fig.suptitle('Output of layer{} (Given image{})'.format(cnt,choose_id))
            img_path = os.path.join(vis_dir,store_path)
            if not os.path.isdir(img_path):
                os.mkdir(img_path)
            fig.savefig(os.path.join(img_path,'layer{}'.format(cnt)))

    else:
        name_ls = ['conv2d_9']
        collect_layers = list()
        collect_layers.append(layer_dict[name_ls[0]].output)

        for cnt, c in enumerate(collect_layers):
            filter_imgs = [[] for i in range(NUM_STEPS//RECORD_FREQ)]
            nb_filter = c.shape[-1]
            for it in range(NUM_STEPS//RECORD_FREQ):
                for filter_idx in range(nb_filter):
                    input_img_data = np.random.random((1, 48, 48, 1))
                    loss = K.mean(c[:,:,:,filter_idx])
                    grads = normalize(K.gradients(loss,input_img)[0])
                    iterate = K.function([input_img],[loss,grads])

                    filter_imgs[it].append(grad_ascent(NUM_STEPS, input_img_data, iterate))
                print(it)

            for it in range(NUM_STEPS//RECORD_FREQ):
                fig = plt.figure(figsize=(14,8))
                for i in range(nb_filter):
                    ax = fig.add_subplot(int(nb_filter)/16,16,i+1)
                    print(filter_imgs[it][i][0])
                    ax.imshow(filter_imgs[it][i][0],cmap='Purples')
                    plt.xticks(np.array([]))
                    plt.yticks(np.array([]))
                    plt.xlabel('{:.3f}'.format(filter_imgs[it][i][1]))
                    plt.tight_layout()
                fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[0],it*RECORD_FREQ))
                img_path = os.path.join(filter_dir,'{}-{}'.format(store_path,name_ls[0]))
                if not os.path.isdir(img_path):
                    os.mkdir(img_path)
                fig.savefig(os.path.join(img_path,'e{}'.format(it*RECORD_FREQ)))

if __name__ == "__main__":
    main()
