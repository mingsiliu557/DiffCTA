import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

from image_adapt.guided_diffusion import dist_util, logger
from image_adapt.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from image_adapt.guided_diffusion.image_datasets import load_data, ImageDataset
from torchvision import utils
import math
from torch.nn.parallel.distributed import DistributedDataParallel as DDP


# added
def load_reference(data_dir, batch_size, image_size, class_cond=False,):
    data_loader = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
        deterministic=True,
        random_flip=False,
        #corruption=corruption,
        #severity=severity,
    )

    return data_loader


def main():
    args = create_argparser().parse_args()

    th.manual_seed(42)

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    logger.log("creating model...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    #print(args.model_path)
    #model.load_state_dict(
    #    dist_util.load_state_dict(args.model_path, map_location="cpu")
    #)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()

    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )
    model.eval()

    logger.log("creating resizers...")
    assert math.log(args.D, 2).is_integer()

    logger.log("loading data...")
    dataloader = load_reference(
        args.base_samples,
        args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    #assert args.num_samples >= args.batch_size * dist_util.get_world_size(), "The number of the generated samples will be larger than the specified number."
    

    logger.log("creating samples...")
    count = 0
    #while count * args.batch_size * dist_util.get_world_size() < args.num_samples:
    for model_kwargs, filename in dataloader:
        #logger.log(f"Generating batch {count + 1}")
        #model_kwargs, filename = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            noise=model_kwargs["ref_img"],
            N=args.N,
            D=args.D,
            scale=args.scale
        )

        for i in range(args.batch_size):
            #path = os.path.join(logger.get_dir(), filename[i].split('/')[0])
            #os.makedirs(path, exist_ok=True)
            out_path = os.path.join(logger.get_dir(), filename[i])

            utils.save_image(
                sample[i].unsqueeze(0),
                out_path,
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )
            logger.log(f"Saved sample to: {out_path}")
        count += 1
        logger.log(f"created {count * args.batch_size * dist_util.get_world_size()} samples")

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=4,
        D=4, # scaling factor
        N=50,
        scale=6,
        use_ddim=False,
        base_samples="",
        model_path="",
        save_dir="",
        save_latents=False,
       # source_dataset="", # One of the dataset:   ['RIM_ONE_r3', 'REFUGE', 'Drishti_GS', 'ORIGA', 'Retina']
        #target_dataset=['RIM_ONE_r3', 'REFUGE', 'Drishti_GS', 'ORIGA', 'Retina']
        #corruption="shot_noise",
        #severity=5,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()