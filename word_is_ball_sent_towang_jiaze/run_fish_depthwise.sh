#!/bin/bash
# need conda activate mask , if dont hava, just install from environment.yaml
CUDA_VISIBLE_DEVICES='0'
python main.py --train_id fish_depthwise \
    --classifer train  \
    --data_path /home/liuyang/workspace/data/classif__fish_11_原件勿动/label/train.csv  \
    --val_path /home/liuyang/workspace/data/classif__fish_11_原件勿动/label/validation.txt \
    --mode_path models \
    --input_size 256 \
    --epochs 5000 \
    --batch_size 64 \
    --workers 6 \
    --lr 0.001 \
    --num_class 9 \
    --model_name ball_depthwise_net \
    #--pretrain True \
    #--pretrain_epoch 50 \
