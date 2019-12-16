#!/bin/bash

python train.py --cfg=experiments/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml --testModel=model_states/hrnetv2_w64_imagenet_pretrained.pth --dataDir=../../data --modelFamily=hrnet --transferEpochs=25 --transferBatchSize=20 > hrnet-transfer-largest-epochs25.log

python train.py --cfg=experiments/cls_resnet152.yaml --dataDir=../../data --modelFamily=resnet152 --transferEpochs=25 --transferBatchSize=32 > resnet152-transfer-epochs25.log

python train.py --cfg=experiments/cls_hrnet_w18_small_v1_sgd_lr5e-2_wd1e-1_bs32_x100.yaml --testModel=model_states/hrnet_w18_small_model_v1.pth --dataDir=../../data --modelFamily=hrnet --transferEpochs=25 --transferBatchSize=128  > hrnet-transfer-smallest-epochs25.log

python train.py --cfg=experiments/cls_resnet18.yaml --dataDir=../../data --modelFamily=resnet18 --transferEpochs=25 --transferBatchSize=256 > resnet18-transfer-epochs25.log
