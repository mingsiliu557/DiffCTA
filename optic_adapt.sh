export PYTHONPATH=$PYTHONPATH:$(pwd)

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

CUDA_VISIBLE_DEVICES=2 mpiexec -n 1 python image_adapt/scripts/optic_sample.py $MODEL_FLAGS \
                            --batch_size 1 --num_samples 1000 --timestep_respacing 100 \
                            --model_path /data1/zhangxingshuai/lms/diffusion_training/ckpt/idrid --base_samples data \
                      --source_dataset idrid