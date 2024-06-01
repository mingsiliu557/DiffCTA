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

## Pre-trained Models
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



* **分类任务**
```
CUDA_VISIBLE_DEVICES=0 python OPTIC/train_source.py --Source_Dataset RIM_ONE_r3 --path_save_log OPTIC/logs --path_save_model OPTIC/models --dataset_root your_dataset_root
```

## How to Run
Please first modify the root in ```VPTTA_OPTIC.sh``` , and then run the following command to reproduce the results.
```
# Reproduce the results on the OD/OC segmentation task
bash VPTTA_OPTIC.sh
# Reproduce the results on the polyp segmentation task
bash VPTTA_POLYP.sh
```

## Citation ✏️
If this code is helpful for your research, please cite:
```
@article{chen2023vptta,
  title={Each Test Image Deserves A Specific Prompt: Continual Test-Time Adaptation for 2D Medical Image Segmentation},
  author={Chen, Ziyang and Ye, Yiwen and Lu, Mengkang and Pan, Yongsheng and Xia, Yong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={},
  year={2024}
}
```
## Acknowledgement
Parts of the code are based on the Pytorch implementations of [DoCR](https://github.com/ShishuaiHu/DoCR), [DLTTA](https://github.com/med-air/DLTTA), and [DomainAdaptor](https://github.com/koncle/DomainAdaptor).

## Contact
Ziyang Chen (zychen@mail.nwpu.edu.cn)
