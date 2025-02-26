#!/bin/bash

#Please modify the following roots to yours.
dataset_root=/data1/zhangxingshuai/lms/OPTIC_CLASSIFY/data
model_root=/data1/zhangxingshuai/lms/OPTIC_CLASSIFY/Diabetic/models/
path_save_log=/data1/zhangxingshuai/lms/OPTIC_CLASSIFY/Diabetic/logs/
generate_root=/data1/zhangxingshuai/lms/OPTIC_CLASSIFY/generated

#Dataset [RIM_ONE_r3, REFUGE, ORIGA, ACRIMA, Drishti_GS]
Source=SYSU
Target_Dataset="['aptos2019','messidor2','idrid', 'SYSU']"

#Optimizer
optimizer=Adam
lr=0.05

#Hyperparameters
memory_size=40
neighbor=16
prompt_alpha=0.01
warm_n=5

#Command
cd Diabetic
CUDA_VISIBLE_DEVICES=4 python cotta_tta.py \
--dataset_root $dataset_root --model_root $model_root --path_save_log $path_save_log \
--Source_Dataset $Source \
--optimizer $optimizer --lr $lr \
--memory_size $memory_size --neighbor $neighbor --prompt_alpha $prompt_alpha --warm_n $warm_n \
--generate_root $generate_root --Target_Dataset "$Target_Dataset"