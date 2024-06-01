import numpy as np
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, ContrastAugmentationTransform, FancyColorTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
import albumentations as albu
import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode, functional as F
import random
import numpy as np
import cv2

class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), p=0.5):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = np.array(img)
            if len(img.shape) == 2:  # Grayscale image
                img = cv2.createCLAHE(self.clip_limit, self.tile_grid_size).apply(img)
            else:  # RGB image
                lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                lab[..., 0] = cv2.createCLAHE(self.clip_limit, self.tile_grid_size).apply(lab[..., 0])
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            img = F.to_pil_image(img)
        return img

class GaussianNoiseTransform:
    def __init__(self, mean=0.0, std=1.0, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = np.array(img).astype(np.float32)
            noise = np.random.normal(self.mean, self.std, img.shape)
            img = np.clip(img + noise * 255, 0, 255).astype(np.uint8)
            img = F.to_pil_image(img)
        return img

class SharpenTransform:
    def __init__(self, alpha=1.5, p=0.5):
        self.alpha = alpha
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = np.array(img)
            blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
            img = cv2.addWeighted(img, self.alpha, blurred, -self.alpha + 1, 0)
            img = F.to_pil_image(img)
        return img


class MotionBlurTransform:
    def __init__(self, kernel_size=3, angle=0, p=0.5):
        self.kernel_size = kernel_size
        self.angle = angle
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = np.array(img)
            kernel = np.zeros((self.kernel_size, self.kernel_size))
            kernel[int((self.kernel_size - 1) / 2), :] = np.ones(self.kernel_size)
            kernel = cv2.warpAffine(
                kernel,
                cv2.getRotationMatrix2D((self.kernel_size / 2 - 0.5, self.kernel_size / 2 - 0.5), self.angle, 1.0),
                (self.kernel_size, self.kernel_size)
            )
            kernel /= np.sum(kernel)
            img = cv2.filter2D(img, -1, kernel)
            img = F.to_pil_image(img)
        return img

class RandomGammaTransform:
    def __init__(self, gamma_range=(0.5, 2.0), p=0.5):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = torch.from_numpy(img).float() / 255.0
            gamma = random.uniform(*self.gamma_range)
            img = torch.pow(img, gamma)
            img = (img * 255).byte().numpy()  
        return img
def get_training_augmentation_torch():
    train_transforms = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([
            T.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.8, 1.2),
                interpolation=InterpolationMode.BILINEAR,
                fill=0
            )
        ], p=1),
        GaussianNoiseTransform(mean=0.0, std=0.05, p=0.2),
        T.RandomChoice([
            CLAHETransform(p=1),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            RandomGammaTransform(gamma_range=(0.5, 2.0), p=0.5)
        ]),
        T.RandomChoice([
            SharpenTransform(alpha=1.5, p=1),
            T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            MotionBlurTransform(kernel_size=3, angle=15, p=1)
        ]),
        T.RandomChoice([
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ColorJitter(hue=0.2, saturation=0.2)
        ]),
    ])
    return train_transforms

# transform = get_training_augmentation_torch()
# transformed_image = transform(image)

def get_train_transform(patch_size=(512, 512)):
    tr_transforms = []
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15))
    # tr_transforms.append(ContrastAugmentationTransform((0.75, 1.25), per_channel=True, p_per_sample=0.15))
    # tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))
    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))
    # tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True,
    #                                            p_per_channel=0.5, p_per_sample=0.15))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

def get_training_augmentation():
	train_transform = [

		albu.HorizontalFlip(p=0.5),
		# albu.ElasticTransform(),
		albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
		
		# albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
		# albu.RandomResizedCrop(height=512, width=512, scale=(0.8,1), ratio=(0.99,1.01),always_apply=True),
        #albu.Resize(height=512, width=512,always_apply=True),

		albu.GaussNoise(p=0.2),
		# albu.IAAPerspective(p=0.5),

		albu.OneOf(
			[
				albu.CLAHE(p=1),
				albu.RandomBrightnessContrast(p=1),
				albu.RandomGamma(p=1),
			],
			p=0.9,
		),

		albu.OneOf(
			[
				albu.Sharpen(p=1),
				albu.Blur(blur_limit=3, p=1),
				albu.MotionBlur(blur_limit=3, p=1),
			],
			p=0.9,
		),

		albu.OneOf(
			[
				albu.RandomBrightnessContrast(p=1),
				albu.HueSaturationValue(p=1),
			],
			p=0.9,
		),
	]
	return albu.Compose(train_transform)

def collate_fn_w_transform(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    #data_dict = {'data': image, 'mask': label, 'name': name}
    data_dict = {'data': image, 'cls': label, 'name': name}
    tr_transforms = get_training_augmentation()
    #tr_transforms = get_training_augmentation_torch()
    aug_data = tr_transforms(imag=data_dict['data'])
    data_dict['data'] = aug_data
    #data_dict['mask'] = to_one_hot_list(data_dict['mask'])
    return data_dict



def collate_fn_wo_transform(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'mask': label, 'name': name}
    data_dict['mask'] = to_one_hot_list(data_dict['mask'])
    return data_dict


def to_one_hot_list(mask_list):
    list = []
    for i in range(mask_list.shape[0]):
        mask = to_one_hot(mask_list[i].squeeze(0))
        list.append(mask)
    return np.stack(list, 0)


def to_one_hot(pre_mask, classes=2):
    mask = np.zeros((pre_mask.shape[0], pre_mask.shape[1], classes))
    mask[pre_mask == 1] = [1, 0]
    mask[pre_mask == 2] = [1, 1]
    return mask.transpose(2, 0, 1)

