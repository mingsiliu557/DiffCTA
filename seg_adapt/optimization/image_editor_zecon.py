#from pathlib import Path

# from optimization.augmentations import ImageAugmentations as ImageAugmentations
import torch
from torchvision import transforms
import torch.nn.functional as F
#from optimization.losses import range_loss, d_clip_loss, d_clip_dir_loss, mse_loss, get_features, zecon_loss_direct
from torch import nn
import kornia.augmentation as K
import open_clip
from types import SimpleNamespace


class ImageAugmentations(nn.Module):
    def __init__(self, output_size, aug_prob, p_min, p_max, patch=False):
        super().__init__()
        self.output_size = output_size
        
        self.aug_prob = aug_prob
        self.patch = patch
        self.augmentations = nn.Sequential(
            K.RandomAffine(degrees=15, translate=0.1, p=aug_prob, padding_mode="border"),  # type: ignore
            K.RandomPerspective(0.7, p=aug_prob),
        )
        self.random_patch = K.RandomResizedCrop(size=(128,128), scale=(p_min,p_max))
        self.avg_pool = nn.AdaptiveAvgPool2d((self.output_size, self.output_size))

    def forward(self, input, num_patch=32, is_global=False):
        """Extents the input batch with augmentations

        If the input is consists of images [I1, I2] the extended augmented output
        will be [I1_resized, I2_resized, I1_aug1, I2_aug1, I1_aug2, I2_aug2 ...]

        Args:
            input ([type]): input batch of shape [batch, C, H, W]

        Returns:
            updated batch: of shape [batch * augmentations_number, C, H, W]
        """
        if self.patch:
            if is_global:
                input = input.repeat(num_patch,1,1,1)
            else:
                input_patches = []
                for i in range(num_patch):
                    if self.aug_prob > 0.0:
                        tmp = self.augmentations(self.random_patch(input))
                    else:
                        tmp = self.random_patch(input)
                    input_patches.append(tmp)
                input = torch.cat(input_patches,dim=0)
        
        else:
            input_patches = []
            for i in range(num_patch):
                input_patches.append(self.augmentations(input))
            input = torch.cat(input_patches,dim=0)
        
        resized_images = self.avg_pool(input)
        return resized_images

def d_clip_loss(x, y, use_cosine=False):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    if use_cosine:
        distance = 1 - (x @ y.t()).squeeze()
    else:
        distance = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

    return distance


class ImageEditor:
    def __init__(self, args) -> None:

        
        # self.clip_model = (
        #     clip.load("ViT-B/16", device=self.device, jit=False)[0].eval().requires_grad_(False)
        # )
        self.clip_model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.args = args
        self.device = args.device
        self.tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.clip_model.to(self.device).eval()

        # self.clip_size = self.clip_model.visual.input_resolution
        self.clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )
        
        self.image_augmentations = ImageAugmentations(224, aug_prob=1, p_min=0.01, p_max=0.3, patch=False)

    
    def clip_global_loss(self,x_in,text):
        # text_embed = self.clip_model.encode_text(
        #     self.clip_model.tokenize(text).to(self.device)
        # ).float()
        text_inputs = self.tokenizer([text]).to(self.device)
        text_embed = self.clip_model.encode_text(text_inputs).float()
        clip_loss = torch.tensor(0)
        augmented_input = self.image_augmentations(x_in,num_patch=self.args.n_patch).add(1).div(2)
        # clip_in = self.clip_normalize(augmented_input)
        #clip_in = self.preprocess_val(augmented_input).to(self.device)
        clip_in = self.clip_normalize(augmented_input).to(self.device)
        image_embeds = self.clip_model.encode_image(clip_in).float()
        dists = d_clip_loss(image_embeds, text_embed)
        for i in range(self.args.batch_size):
            clip_loss = clip_loss + dists[i :: self.args.batch_size].mean()

        return clip_loss

def main():
    # 设置参数
    args = SimpleNamespace(
        aug_prob=0.8,      
        p_min=0.01,         
        p_max=0.3,          
        n_patch=32,         
        batch_size=1      
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    editor = ImageEditor(args)
    editor.device = device

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor()
    # ])
 
    x_in = torch.randn(args.batch_size, 3, 256, 256).to(device) 


    text = "a photo of a retinal fundus"

 
    clip_loss = editor.clip_global_loss(x_in, text)
    print("CLIP Loss:", clip_loss.item())

if __name__ == "__main__":
    main()

