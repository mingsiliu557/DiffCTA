import sys
sys.path.append('/home/lmx/VPTTA/OPTIC')
import math
from torch.utils import data
import numpy as np
from PIL import Image
from batchgenerators.utilities.file_and_folder_operations import *
from dataloaders.normalize import normalize_image, normalize_image_to_0_1
from dataloaders.convert_csv_to_list import convert_labeled_list
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloaders.transform import collate_fn_w_transform, collate_fn_wo_transform



class OPTIC_dataset(data.Dataset):
    def __init__(self, root, generate_root, img_list, label_list, target_size=512, batch_size=None, img_normalize=True):
        super().__init__()
        self.root = root
        self.generate_root = generate_root
        self.img_list = img_list
        self.label_list = label_list
        self.len = len(img_list)
        self.target_size = (target_size, target_size)
        self.img_normalize = img_normalize
        # if batch_size is not None:
        #     iter_nums = len(self.img_list) // batch_size
        #     scale = math.ceil(100 / iter_nums)
        #     self.img_list = self.img_list * scale
        #     self.label_list = self.label_list * scale

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        if self.label_list[item].endswith('tif'):
            self.label_list[item] = self.label_list[item].replace('.tif', '-{}.tif'.format(1))
        img_file = os.path.join(self.root, self.img_list[item])
        g_img_file = os.path.join(self.generate_root, self.img_list[item])
        label_file = os.path.join(self.root, self.label_list[item])
        img = Image.open(img_file)
        g_img = Image.open(g_img_file)
        label = Image.open(label_file).convert('L')

        img = img.resize(self.target_size)
        g_img = g_img.resize(self.target_size)
        label = label.resize(self.target_size, resample=Image.NEAREST)
        img_npy = np.array(img).transpose(2, 0, 1).astype(np.float32)
        g_img_npy = np.array(g_img).transpose(2, 0, 1).astype(np.float32)
        if self.img_normalize:
            img_npy = normalize_image_to_0_1(img_npy)
            g_img_npy = normalize_image_to_0_1(g_img_npy)
        label_npy = np.array(label)

        mask = np.zeros_like(label_npy)
        mask[label_npy < 255] = 1
        mask[label_npy == 0] = 2
        return img_npy, g_img_npy, mask[np.newaxis], img_file

class RIM_ONE_dataset(data.Dataset):
    def __init__(self, root, img_list, label_list, target_size=256, batch_size=None, img_normalize=True):
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.label_list = label_list
        self.len = len(img_list)
        self.target_size = (target_size, target_size)
        self.img_normalize = img_normalize

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_file = os.path.join(self.root, self.img_list[item])
        try:
            img = Image.open(img_file)
            if img is None or img.size == 0:
                raise ValueError(f"Image is empty or corrupt: {img_file}")
            img = img.resize(self.target_size)
            img_npy = np.array(img).transpose(2, 0, 1).astype(np.float32)
        except Exception as e:
            print(f"Error loading image {img_file}: {e}")
            return None  # Return None or handle it as per your requirement

        if self.img_normalize:
            img_npy = normalize_image_to_0_1(img_npy)

        filename = os.path.basename(self.img_list[item])
        if filename.startswith('G') or filename.startswith('g'):
            label = 1
        elif filename.startswith('N') or filename.startswith('S') or filename.startswith('n'):
            label = 0
        else:
            label = -1

        return img_npy, label, img_file

class Ensemble_dataset(data.Dataset):
    def __init__(self, root, generate_root, img_list, label_list, target_size=256, batch_size=None, img_normalize=True):
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.label_list = label_list
        self.generate_root = generate_root
        self.len = len(img_list)
        self.target_size = (target_size, target_size)
        self.img_normalize = img_normalize


    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_file = os.path.join(self.root, self.img_list[item])
        generate_file = os.path.join(self.generate_root, self.img_list[item])
        try:
            img = Image.open(img_file)
            if img is None or img.size == 0:
                raise ValueError(f"Image is empty or corrupt: {img_file}")
            img = img.resize(self.target_size)
            img_npy = np.array(img).transpose(2, 0, 1).astype(np.float32)
        except Exception as e:
            print(f"Error loading image {img_file}: {e}")
            return None  # Return None or handle it as per your requirement
        try:
            g_img = Image.open(generate_file)
            if img is None or g_img.size == 0:
                raise ValueError(f"Image is empty or corrupt: {img_file}")
            g_img = g_img.resize(self.target_size)
            g_img_npy = np.array(g_img).transpose(2, 0, 1).astype(np.float32)
        except Exception as e:
            print(f"Error loading image {img_file}: {e}")
            return None  # Return None or handle it as per your requirement
        if self.img_normalize:
            img_npy = normalize_image_to_0_1(img_npy)
            g_img_npy = normalize_image_to_0_1(g_img_npy)
        filename = os.path.basename(self.img_list[item])
        if filename.startswith('G') or filename.startswith('g'):
            label = 1
        elif filename.startswith('N') or filename.startswith('S') or filename.startswith('n'):
            label = 0
        else:
            label = -1

        return img_npy, g_img_npy, label, img_file

# def main():
#     source_train_csv = []
#     #source_train_csv.append('REFUGE' + '_train.csv')
#     source_train_csv.append('REFUGE' + '_test.csv')
#     sr_img_list, sr_label_list = convert_labeled_list('/home/lmx/VPTTA/Data', source_train_csv)
#     train_dataset = Ensemble_dataset('/home/lmx/VPTTA/Data', '/home/lmx/VPTTA/generated', sr_img_list, sr_label_list, 256, img_normalize=False, batch_size=1)
#     source_train_loader = DataLoader(dataset=train_dataset,
#                                               batch_size=1,
#                                               shuffle=True,
#                                               pin_memory=True,
#                                               collate_fn=collate_fn_wo_transform,
#                                               num_workers=1)
    
#     for batch, data in enumerate(source_train_loader):
#         x, x_g, y = data['data'], data['g_data'] , data['cls']
#     print('load data successfully')
# if __name__ == '__main__':
#     main()