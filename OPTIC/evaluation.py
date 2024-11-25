import os
import torch
import numpy as np
import argparse, sys, datetime
import torch.nn.functional as F
from config import Logger
from torch.autograd import Variable
from utils.convert import AdaBN
from utils.memory import Memory
from utils.prompt import Prompt
from utils.metrics import calculate_metrics, calculate_cls_metrics
from networks.resnet import resnet50, resnet18
from torch.utils.data import DataLoader
from dataloaders.OPTIC_dataloader import OPTIC_dataset, RIM_ONE_dataset, Ensemble_dataset
from dataloaders.transform import collate_fn_wo_transform
from dataloaders.convert_csv_to_list import convert_labeled_list
from dataloaders.normalize import normalize_image, normalize_image_to_0_1, normalize_image_to_imagenet_standards
import ast

torch.set_num_threads(1)

def adjust_predictions_with_temperature_scaling(logits, temperature=1.5):
    """
    Adjust predictions to minimize entropy using temperature scaling.
    This method scales the logits before applying softmax, which can
    effectively control the confidence of the predictions without distorting the distribution.
    
    Parameters:
        logits (Tensor): The logits from a model's output.
        temperature (float): The temperature factor to scale the logits. Higher values produce softer probabilities.

    Returns:
        Tensor: Adjusted probabilities that are softer, reducing the entropy of the distribution.
    """
    scaled_logits = logits / temperature
    adjusted_probs = F.softmax(scaled_logits, dim=1)
    return adjusted_probs

def fuse_model(model):
    """
    This function will traverse the model and automatically fuse Convolution and BatchNorm layers.
    """
    import torch.nn as nn
    for module_name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            next_name, next_module = next(model.named_children())
            if isinstance(next_module, nn.BatchNorm2d):
                # Fuse this conv and batch norm
                torch.quantization.fuse_modules(model, [module_name, next_name], inplace=True)
        else:
            # Recursively apply to sub-modules
            fuse_model(module)

class VPTTA:
    def __init__(self, config):
        # Save Log
        time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")
        log_root = os.path.join(config.path_save_log, 'VPTTA')
        if not os.path.exists(log_root):
            os.makedirs(log_root)
        log_path = os.path.join(log_root, time_now + '.log')
        sys.stdout = Logger(log_path, sys.stdout)

        # Data Loading
        target_test_csv = []
        for target in config.Target_Dataset:
            if target != 'REFUGE_Valid':
                target_test_csv.append(target + '_train.csv')
                target_test_csv.append(target + '_test.csv')
            else:
                target_test_csv.append(target + '.csv')
        ts_img_list, ts_label_list = convert_labeled_list(config.dataset_root, target_test_csv)
        target_test_dataset = RIM_ONE_dataset(config.dataset_root, ts_img_list, ts_label_list,
                                            config.image_size, img_normalize=False)
        self.target_test_loader = DataLoader(dataset=target_test_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False,
                                             collate_fn=collate_fn_wo_transform,
                                             num_workers=config.num_workers)

        # Model
        self.load_model = os.path.join(config.model_root, str(config.Source_Dataset))  # Pre-trained Source Model
        self.backbone = config.backbone
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch

        # GPU
        self.device = config.device
        
        self.quant = torch.quantization.QuantStub()

        # Warm-up
        self.warm_n = config.warm_n

        # Initialize the pre-trained model
        self.build_model()

        # Print Information
        for arg, value in vars(config).items():
            print(f"{arg}: {value}")
        print('***' * 20)

    def build_model(self):
        #self.model = ResUnet(resnet=self.backbone, num_classes=self.out_ch, pretrained=False, newBN=AdaBN, warm_n=self.warm_n).to(self.device)
        self.model = resnet18(pretrained= False, num_classes=self.out_ch)
        checkpoint = torch.load(os.path.join(self.load_model, 'last-Resnet18.pth'))
        # self.model.to('cpu') 
        # checkpoint = torch.load(os.path.join(self.load_model, 'quantized_ResUnet.pth'))
        # fuse_model(self.model)  # Fuse Conv, BN, etc. as per your model's fusion setup
        # self.model.qconfig = torch.quantization.default_qconfig  # Match the config settings
        # torch.quantization.prepare(self.model, inplace=True)
        # torch.quantization.convert(self.model, inplace=True)
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.to(self.device)
        # self.model.to('cuda') 

    def run(self):
        #metric_dict = ['Disc_Dice', 'Disc_ASD', 'Cup_Dice', 'Cup_ASD']

        # Valid on Target
        metrics_test = [[]]
        metric_dict = ['Acc']

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)

        for batch, data in enumerate(self.target_test_loader):
            x, y = data['data'], data['cls']
            x = torch.from_numpy(normalize_image_to_imagenet_standards(x)).to(dtype=torch.float32)
            y = torch.from_numpy(y).to(dtype=torch.long)

            x, y = Variable(x).to(self.device), Variable(y).to(self.device)
            self.model.eval()

            with torch.no_grad():
                x = self.quant(x)
                pred_logit = self.model(x)
            # adjusted_probs = adjust_predictions_with_temperature_scaling(pred_logit)

            # Calculate the metrics
            #seg_output = torch.sigmoid(pred_logit)
            metrics = calculate_cls_metrics(pred_logit.detach().cpu(), y.detach().cpu())
            for i in range(len(metrics)):
                #assert isinstance(metrics[i], list), "The metrics value is not list type."
                #metrics_test[i] += metrics[i]
                metrics_test[i].append(metrics[i])

        test_metrics_y = np.mean(metrics_test, axis=1)
        print_test_metric_mean = {}
        for i in range(len(test_metrics_y)):
            print_test_metric_mean[metric_dict[i]] = test_metrics_y[i]
        print("Test Metrics: ", print_test_metric_mean)
        #print('Mean Acc:', (print_test_metric_mean['Disc_Dice'] + print_test_metric_mean['Cup_Dice']) / 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--Source_Dataset', type=str, default='ORIGA',
                        help='RIM_ONE_r3/REFUGE/ORIGA/REFUGE_Valid/Drishti_GS')
    parser.add_argument('--Target_Dataset', type=str)

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=256)

    # Model
    parser.add_argument('--backbone', type=str, default='resnet18', help='resnet34/resnet50')
    parser.add_argument('--in_ch', type=int, default=3)
    parser.add_argument('--out_ch', type=int, default=2)

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', help='SGD/Adam')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.99)  # momentum in SGD
    parser.add_argument('--beta1', type=float, default=0.9)      # beta1 in Adam
    parser.add_argument('--beta2', type=float, default=0.99)     # beta2 in Adam.
    parser.add_argument('--weight_decay', type=float, default=0.00)

    # Training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--iters', type=int, default=1)

    # Hyperparameters in memory bank, prompt, and warm-up statistics
    parser.add_argument('--memory_size', type=int, default=40)
    parser.add_argument('--neighbor', type=int, default=16)
    parser.add_argument('--prompt_alpha', type=float, default=0.01)
    parser.add_argument('--warm_n', type=int, default=5)

    # Path
    parser.add_argument('--path_save_log', type=str, default='./logs')
    parser.add_argument('--model_root', type=str, default='./OPTIC/models')
    parser.add_argument('--dataset_root', type=str, default='/home/lmx/VPTTA/Data')
    parser.add_argument('--generate_root', type=str, default='/home/lmx/VPTTA/generated')

    # Cuda (default: the first available device)
    parser.add_argument('--device', type=str, default='cuda:0')

    config = parser.parse_args()

    config.Target_Dataset = ast.literal_eval(config.Target_Dataset)
    config.Target_Dataset.remove(config.Source_Dataset)

    TTA = VPTTA(config)
    TTA.run()
