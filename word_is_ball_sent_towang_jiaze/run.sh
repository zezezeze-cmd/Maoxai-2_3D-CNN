#!/bin/bash
# need conda activate mask , if dont hava, just install from environment.yaml
CUDA_VISIBLE_DEVICES='0'
python main.py --train_id imgnet_depthwise_net_0001 \
    --classifer train  \
    --data_path ~/DataPublic/train/train_shuffle.csv  \
    --mode_path models \
    --input_size 224 \
    --epochs 5000 \
    --batch_size 64 \
    --num_class 1000 \
    --workers 6 \
    --lr 0.001 \
    --model_name ball_depthwise_net \
    #--pretrain True \
    #--pretrain_epoch 50 \
