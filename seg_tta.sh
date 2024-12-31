#!/bin/bash

#Please modify the following roots to yours.
dataset_root=/lmx/data/OPTIC_CLASSIFY/Data
generate_root=/lmx/data/OPTIC_CLASSIFY/seg_generated
model_root=/lmx/data/OPTIC_CLASSIFY/OPTIC_SEG/models/
path_save_log=/lmx/data/OPTIC_CLASSIFY/OPTIC_SEG/logs/

#Dataset [BKAI, CVC-ClinicDB, ETIS-LaribPolypDB, Kvasir-SEG]
Source=Drishti_GS

#Optimizer
optimizer=Adam
lr=0.01

#Hyperparameters
memory_size=40
neighbor=16
prompt_alpha=0.01
warm_n=5

#Command
cd OPTIC_SEG
CUDA_VISIBLE_DEVICES=0 python vptta.py \
--dataset_root $dataset_root --model_root $model_root --path_save_log $path_save_log --generate_root $generate_root \
--Source_Dataset $Source \
--optimizer $optimizer --lr $lr \
--memory_size $memory_size --neighbor $neighbor --prompt_alpha $prompt_alpha --warm_n $warm_n