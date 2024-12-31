import numpy as np
import cv2
import albumentations as A
import  torch

class TTA_2d():
    def __init__(self, flip=False, rotate=False, aug_a=None, aug_b=None):
        self.flip = flip
        self.rotate = rotate
        self.aug_a = aug_a
        self.aug_b = aug_b
        self.params = []  # 用于存储每次增强的参数

    def img_list(self, img):
        # for ISIC, the shape is torch.size(b, c, h, w)
        img = img.detach().cpu().numpy()
        out = []
        self.params = []# 清空每一次参数
        out.append(img)

        # GTAug-A 的像素级增强
        if self.aug_a:
            out = [self.aug_a.augment_image(x) for x in out]  # 每张图像都做增强

        if self.flip:
            # apply flip
            for i in range(2,4):
                out.append(np.flip(img, axis=i))
        if self.rotate:
            # apply rotation
            for i in range(1, 4):
                out.append(np.rot90(img, k=i, axes=(2,3)))

        # GTAug-B 的变换
        if self.aug_b:
            # augmented_out = []
            # for image in out:
            augmented = self.aug_b.augment(image=img)
            out.append(augmented["image"])
            
            
            self.params.append(augmented["params"])  # 记录变换参数
            # out = augmented_out
            
        out = [torch.from_numpy(x.copy()).float() for x in out]
    
        return out
    
    def img_list_inverse(self, img_list):
        # for ISIC, the shape is numpy(b, h, w)
        img_list = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in img_list]

        out = [img_list[0]]
        if self.flip:
            # Apply flip
            for i in range(2):
                out.append(np.flip(img_list[i + 1], axis=i + 1))  # Flip along height and width

        if self.rotate:
            # Apply rotation
            for i in range(3):
                out.append(np.rot90(img_list[i + 3], k=-(i + 1), axes=(2, 3)))  # Rotate along height and width

        if self.aug_b:
            recovered_out = []
            for i, image in enumerate(out):
                if isinstance(self.params[i], dict):  # Check if there are GTAug-B parameters
                    recovered = self.aug_b.recover(image, self.params[i])
                    recovered_out.append(recovered)
                else:
                    recovered_out.append(image)
            out = recovered_out
        
        out = [torch.from_numpy(x.copy()).float() for x in out]

        return out
    




class GTAug_B_2D:
    def __init__(self):
        self.augment = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ])

    def augment_image(self, image):
        augmented = self.augment(image=image)
        return augmented['image']
    
    
    def recover(self, image, params):
        """
        根据记录的参数逆转增强操作。
        :param image: 增强后的图像
        :param params: 增强时记录的参数
        :return: 恢复后的图像
        """
        # Albumentations 暂不支持自动逆变换，需要手动实现
        recovered = image
        # 恢复 Shift
        if "dx" in params and "dy" in params:
            dx, dy = params["dx"], params["dy"]
            recovered = np.roll(recovered, shift=(-dx, -dy), axis=(1, 2))

        # 恢复 Scale (需要记录原始尺寸并缩放回来)
        if "scale" in params:
            scale = params["scale"]
            height, width = recovered.shape[1:3]
            new_h, new_w = int(height / scale), int(width / scale)
            recovered = cv2.resize(recovered, (new_w, new_h))

        # 恢复 Rotate
        if "angle" in params:
            angle = params["angle"]
            recovered = A.geometric.rotate.replay_compose_rotate(
                recovered, angle=-angle, interpolation=cv2.INTER_LINEAR
            )

        return recovered



class GTAug_A_2D:
    def __init__(self):
        self.augment = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            #A.CLAHE(p=0.5)
        ])

    def augment_image(self, image):
        augmented = self.augment(image=image)
        return augmented['image']

# augmenter_a = GTAug_A_2D()
# augmented_image_a = augmenter_a.augment_image(image)


import imp
import skimage.morphology as morph
import numpy as np
from scipy.ndimage import label

def abl(image: np.ndarray, for_which_classes: list, volume_per_voxel: float = None,
                                                   minimum_valid_object_size: dict = None):
    """
    removes all but the largest connected component, individually for each class
    :param image:
    :param for_which_classes: can be None. Should be list of int. Can also be something like [(1, 2), 2, 4].
    Here (1, 2) will be treated as a joint region, not individual classes (example LiTS here we can use (1, 2)
    to use all foreground classes together)
    :param minimum_valid_object_size: Only objects larger than minimum_valid_object_size will be removed. Keys in
    minimum_valid_object_size must match entries in for_which_classes
    :return:
    """
    if for_which_classes is None:
        for_which_classes = np.unique(image)
        for_which_classes = for_which_classes[for_which_classes > 0]
    assert 0 not in for_which_classes, "cannot remove background"

    if volume_per_voxel is None:
        volume_per_voxel = 1

    largest_removed = {}
    kept_size = {}
    for c in for_which_classes:
        if isinstance(c, (list, tuple)):
            c = tuple(c)  # otherwise it cant be used as key in the dict
            mask = np.zeros_like(image, dtype=bool)
            for cl in c:
                mask[image == cl] = True
        else:
            mask = image == c
        # get labelmap and number of objects
        lmap, num_objects = label(mask.astype(int))

        # collect object sizes
        object_sizes = {}
        for object_id in range(1, num_objects + 1):
            object_sizes[object_id] = (lmap == object_id).sum() * volume_per_voxel

        largest_removed[c] = None
        kept_size[c] = None

        if num_objects > 0:
            # we always keep the largest object. We could also consider removing the largest object if it is smaller
            # than minimum_valid_object_size in the future but we don't do that now.
            maximum_size = max(object_sizes.values())
            kept_size[c] = maximum_size

            for object_id in range(1, num_objects + 1):
                # we only remove objects that are not the largest
                if object_sizes[object_id] != maximum_size:
                    # we only remove objects that are smaller than minimum_valid_object_size
                    remove = True
                    if minimum_valid_object_size is not None:
                        remove = object_sizes[object_id] < minimum_valid_object_size[c]
                    if remove:
                        image[(lmap == object_id) & mask] = 0
                        if largest_removed[c] is None:
                            largest_removed[c] = object_sizes[object_id]
                        else:
                            largest_removed[c] = max(largest_removed[c], object_sizes[object_id])
    # return image, largest_removed, kept_size
    return image


def rsa(image: np.array, for_which_classes: list, volume_per_voxel: float = None, minimum_valid_object_size: dict = None):
    """
    Remove samll objects, smaller than minimum_valid_object_size, individually for each class
    :param image:
    :param for_which_classes: can be None. Should be list of int. Can also be something like [(1, 2), 2, 4].
    Here (1, 2) will be treated as a joint region, not individual classes (example LiTS here we can use (1, 2)
    to use all foreground classes together)
    :param minimum_valid_object_size: Only objects larger than minimum_valid_object_size will be removed. Keys in
    minimum_valid_object_size must match entries in for_which_classes
    :return:
    """
    if for_which_classes is None:
        for_which_classes = np.unique(image)
        for_which_classes = for_which_classes[for_which_classes > 0]
    assert 0 not in for_which_classes, "cannot remove background"

    if volume_per_voxel is None:
        volume_per_voxel = 1
    
    for c in for_which_classes:
        if isinstance(c, (list, tuple)):
            c = tuple(c)
            mask = np.zeros_like(image, dtype=bool)
            for cl in c:
                mask[image == cl] = True
        else:
            mask = image == c
        # get labelmap and number of objects
        lmap, num_objects = label(mask.astype(int))

        # collect object sizes
        object_sizes = {}
        for object_id in range(1, num_objects + 1):
            object_sizes[object_id] = (lmap == object_id).sum() * volume_per_voxel

        if num_objects > 0:
            # removing the largest object if it is smaller than minimum_valid_object_size.
            for object_id in range(1, num_objects + 1):
                # we only remove objects that are smaller than minimum_valid_object_size
                if object_sizes[object_id] < minimum_valid_object_size[c]:
                    image[(lmap == object_id) & mask] = 0
    
    return image




if __name__ == '__main__':   
    import torch
    aug_a = GTAug_A_2D()
    aug_b = GTAug_B_2D()
    tta = TTA_2d(flip=False, rotate=False, aug_a=None, aug_b = aug_b)
    a = torch.randn(4, 3, 256, 256)
    b = tta.img_list(a)
    print(b[0].shape)
    c = tta.img_list_inverse(b)
    print(c)