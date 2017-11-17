#!/bin/bash
wget https://www.dropbox.com/s/oyhhe2w594ldiot/model_aug_best_65.h5?dl=1 -O model_aug_best_65.h5
wget https://www.dropbox.com/s/qx7ly1qug8qh5c5/model_aug_best_66.h5?dl=1 -O model_aug_best_66.h5
wget https://www.dropbox.com/s/7e6l2b8jvpn6qai/model_aug_best_66731.h5?dl=1 -O model_aug_best_66731.h5
python3 hw3_test.py $1 $2
