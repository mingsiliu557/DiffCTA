#!/bin/bash

#Please modify the following roots to yours.
dataset_root=/home/lmx/VPTTA/Data
model_root=/home/lmx/VPTTA/OPTIC/models/
path_save_log=/home/lmx/VPTTA/OPTIC/logs/
generate_root=/home/lmx/VPTTA/generated

#Dataset [RIM_ONE_r3, REFUGE, ORIGA, ACRIMA, Drishti_GS]
Source=ACRIMA

#Optimizer
optimizer=Adam
lr=0.05

#Hyperparameters
memory_size=40
neighbor=16
prompt_alpha=0.01
warm_n=5

#Command
cd OPTIC
CUDA_VISIBLE_DEVICES=1 python TTA.py \
--dataset_root $dataset_root --model_root $model_root --path_save_log $path_save_log \
--Source_Dataset $Source \
--optimizer $optimizer --lr $lr \
--memory_size $memory_size --neighbor $neighbor --prompt_alpha $prompt_alpha --warm_n $warm_n \
--generate_root $generate_root
