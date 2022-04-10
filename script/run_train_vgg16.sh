#!/bin/bash

# make sure the name of dataset directory is same with `dataset_name`
# ex: /notebooks/dataset/ILSVRC | /notebooks/dataset/CUB

# gpu ids are sepearted with comma. ex: --gpu 0,1,2,3,4,5,6,7 | --gpu 2,5
# dataset_name: (ILSVRC, CUB)
# method_name: ('CAM', 'AAE_SAE')
# arch_name: (resnet50_se, vgg_gap)

python train.py \
      --gpu 1 \
      --epoch 250 \
      --data_dir ./dataset/ \
      --dataset_name CUB \
      --method_name AAE_SAE \
      --arch_name vgg_gap \
      --base_lr 0.01 \
      --log_dir VGG16_CUB_AAE_SAE_2_3_1 \
      --use_pretrained_model \
      --batch 32 \
      --gating_position 3 4 53 \
      --aae_threshold 0.50 \
      --evaluate \

# python train.py \
#       --gpu 0 \
#       --epoch 100 \
#       --data_dir ./dataset/ \
#       --dataset_name ILSVRC \
#       --method_name AAE_SAE \
#       --arch_name vgg_gap \
#       --base_lr 0.01 \
#       --log_dir VGG16_ILSVRC_AAE_SAE_1_0 \
#       --use_pretrained_model \
#       --batch 32 \
#       --gating_position 3 4 53 \
#       --aae_threshold 0.50 \
#       --evaluate \
