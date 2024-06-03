# :page_facing_up: 眼底图像分类


<div align="center">
  
</div>

## Environment
```
环境的配置和以下仓库相同：
https://github.com/shiyegao/DDA
```

## Data Preparation
这里的预处理是对眼底图像做了中心裁剪到800*800
把数据放在根目录下，数据集的组织形式为   根目录/数据集名称/train or test/image/xxx.png or jpg

链接：https://pan.baidu.com/s/1perPGwjWwbqoTHerdFt7aQ?pwd=10v1 
提取码：10v1 
--来自百度网盘超级会员V5的分享

## Pre-trained Models

* **训练源域diffusion模型**
这里需要去预训练一个在源域上的diffusion，训练源域diffusion与guided diffusion仓库一样：
https://github.com/openai/guided-diffusion
这里环境需要多加几个包。同时记得看train_util文件当中的训练逻辑，需要在脚本文件image_train中设置lr_anneal_steps来指明训练的iteration，我这里设置的是100000，大概在4卡4090上跑12 h（这里首先让diffusion加载了256*256 uncond的预训练文件）
    
    
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    #model.to(dist_util.dev())
    model.load_state_dict(torch.load('/home/liumingsi/guided-diffusion/ckpt/256x256_diffusion_uncond.pt', map_location=dist_util.dev()))
    logger.log("load pretrained ckpt successfully!")
    model.to(dist_util.dev())


训练参数：
export PYTHONPATH=$PYTHONPATH:$(pwd)
export OPENAI_LOGDIR=/home/liumingsi/guided-diffusion/log
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
TRAIN_FLAGS="--batch_size 1 --lr 1e-5 --save_interval 10000 --weight_decay 0.05"

CUDA_VISIBLE_DEVICES=0,1 mpiexec -n 2 python scripts/image_train.py --data_dir /home/liumingsi/guided-diffusion/data/acrima $MODEL_FLAGS $TRAIN_FLAGS



* **训练源域模型**
```
CUDA_VISIBLE_DEVICES=0 python OPTIC/train_source.py --Source_Dataset RIM_ONE_r3 --path_save_log OPTIC/logs --path_save_model OPTIC/models --dataset_root your_dataset_root
```

## 生成伪样本
修改optic_adapt.sh脚本（原本的数据集路径，生成的数据集存放路径，source_dataset，model_path等。target_dataset到py里面修改），这里只能用单卡跑，batchsize设为1，因为用多卡跑会出现最后生成的样本丢失几张的情况
```
bash optic_adapt.sh

```
## 无适应测试
修改相应参数，主要为数据集，以及路径等，在bash脚本修改即可
```
bash VPTTA_OPTIC.sh

```

## TTA
修改TTA.sh脚本
```
bash TTA.sh

```

## 实验效果
'''

从单个数据集泛化到其他四个数据集的效果:

RIM_ONE: 57.91->58.18;        REFUGE:42.04--->43.1;          ORIGA:61.98 -->61.7;            ACRIMA: (还在训source pth)：27.62 -->28.15;          Drishti_GS：62.76  -->58.62

'''
