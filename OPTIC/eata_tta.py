import os
from copy import deepcopy
import random
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
from dataloaders.transform import collate_fn_wo_transform_ensemble
from dataloaders.convert_csv_to_list import convert_labeled_list
from dataloaders.normalize import normalize_image, normalize_image_to_0_1, normalize_image_to_imagenet_standards
#import algorithm.cotta.CoTTA as cotta
from algorithm.cotta import CoTTA as cotta
import algorithm.eata as eata
from torchvision.transforms import transforms

torch.set_num_threads(1)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed for reproducibility
set_seed(42)


brightness_change = transforms.ColorJitter(brightness=0.5)

hue_change = transforms.ColorJitter(hue=0.5)

contrast_change = transforms.ColorJitter(contrast=0.5)

color_aug = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)

transform = transforms.Compose([
        brightness_change,
        hue_change,
        contrast_change,
    ])




def adjust_predictions_with_temperature_scaling(logits, temperature=1.5):
    scaled_logits = logits / temperature
    adjusted_probs = F.softmax(scaled_logits, dim=1)
    return adjusted_probs

class VPTTA:
    def __init__(self, config):
        time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")
        log_root = os.path.join(config.path_save_log, 'VPTTA')
        if not os.path.exists(log_root):
            os.makedirs(log_root)
        log_path = os.path.join(log_root, time_now + '.log')
        sys.stdout = Logger(log_path, sys.stdout)

        target_test_csv = []
        for target in config.Target_Dataset:
            if target != 'REFUGE_Valid':
                target_test_csv.append(target + '_train.csv')
                target_test_csv.append(target + '_test.csv')
            else:
                target_test_csv.append(target + '.csv')
        ts_img_list, ts_label_list = convert_labeled_list(config.dataset_root, target_test_csv)
        generate_path = os.path.join(config.generate_root, config.Source_Dataset+'_style')
        target_test_dataset = Ensemble_dataset(config.dataset_root, generate_path, ts_img_list, ts_label_list,
                                               config.image_size, img_normalize=False)
        self.target_test_loader = DataLoader(dataset=target_test_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False,
                                             collate_fn=collate_fn_wo_transform_ensemble,
                                             num_workers=config.num_workers)

        self.load_model = os.path.join(config.model_root, str(config.Source_Dataset))
        self.backbone = config.backbone
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch

        self.device = config.device

        self.quant = torch.quantization.QuantStub()

        self.warm_n = config.warm_n

        self.build_model()

        for arg, value in vars(config).items():
            print(f"{arg}: {value}")
        print('***' * 20)

    def build_model(self):
        self.model = resnet18(pretrained=False, num_classes=self.out_ch)
        checkpoint = torch.load(os.path.join(self.load_model, 'last-Resnet18.pth'))
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def compute_fishers(self, fisher_loader, fisher_size):
        self.model = eata.configure_model(self.model)
        params, param_names = eata.collect_params(self.model)
        ewc_optimizer = torch.optim.SGD(params, 0.001)
        fishers = {}
        train_loss_fn = torch.nn.CrossEntropyLoss().cuda()

        for iter_, data in enumerate(fisher_loader, start=1):
            if iter_ > fisher_size:
                break
            if self.device is not None:
                images, x_g, targets = data['data'], data['g_data'], data['cls']
                images = torch.from_numpy(normalize_image_to_imagenet_standards(images)).to(dtype=torch.float32)
                #x_g = torch.from_numpy(normalize_image_to_imagenet_standards(x_g)).to(dtype=torch.float32)
                targets = torch.from_numpy(targets).to(dtype=torch.long)
                images = images.to(self.device)
                targets = targets.to(self.device)                
            outputs = self.model(images)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if name in fishers:
                        fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter_ == fisher_size:
                        fisher = fisher / iter_
                    fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()

        return fishers    


    def run(self):
        metrics_test = [[]]
        metric_dict = ['Acc']
        fisher_loader = self.target_test_loader  # Assuming the same data loader can be used
        fisher_size = 500  # Or any other number you want
        fishers = self.compute_fishers(fisher_loader, fisher_size)

        for batch, data in enumerate(self.target_test_loader):
            x, x_g, y = data['data'], data['g_data'], data['cls']
            x = torch.from_numpy(normalize_image_to_imagenet_standards(x)).to(dtype=torch.float32)
            x_g = torch.from_numpy(normalize_image_to_imagenet_standards(x_g)).to(dtype=torch.float32)
            y = torch.from_numpy(y).to(dtype=torch.long)

            x, y = Variable(x).to(self.device), Variable(y).to(self.device)
            x_g = Variable(x_g).to(self.device)
            #x = transform(x)
            #x_g = transform(x_g)
            self.model = eata.configure_model(self.model)
            params, param_names = eata.collect_params(self.model)
            self.optimizer = torch.optim.Adam(params, lr=0.00025)
            eata_model = eata.EATA(self.model, self.optimizer, fishers=fishers)


            pred_logit = eata_model(x)


            final_pred_logit = pred_logit
            #final_pred_logit =  pred_logit_g + pred_logit

            metrics = calculate_cls_metrics(final_pred_logit.detach().cpu(), y.detach().cpu())
            for i in range(len(metrics)):
                metrics_test[i].append(metrics[i])

        test_metrics_y = np.mean(metrics_test, axis=1)
        print_test_metric_mean = {}
        for i in range(len(test_metrics_y)):
            print_test_metric_mean[metric_dict[i]] = test_metrics_y[i]
        print("Test Metrics: ", print_test_metric_mean)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Source_Dataset', type=str, default='Drishti_GS', help='RIM_ONE_r3/REFUGE/ORIGA/ACRIMA/Drishti_GS')
    parser.add_argument('--Target_Dataset', type=list)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--backbone', type=str, default='resnet18', help='resnet34/resnet50')
    parser.add_argument('--in_ch', type=int, default=3)
    parser.add_argument('--out_ch', type=int, default=2)
    parser.add_argument('--optimizer', type=str, default='Adam', help='SGD/Adam')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--weight_decay', type=float, default=0.00)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--iters', type=int, default=1)
    parser.add_argument('--memory_size', type=int, default=40)
    parser.add_argument('--neighbor', type=int, default=16)
    parser.add_argument('--prompt_alpha', type=float, default=0.01)
    parser.add_argument('--warm_n', type=int, default=5)
    parser.add_argument('--path_save_log', type=str, default='/lmx/data/OPTIC_CLASSIFY/OPTIC/logs')
    parser.add_argument('--model_root', type=str, default='/lmx/data/OPTIC_CLASSIFY/OPTIC/models')
    parser.add_argument('--dataset_root', type=str, default='/lmx/data/OPTIC_CLASSIFY/Data')
    parser.add_argument('--generate_root', type=str, default='/lmx/data/OPTIC_CLASSIFY/generated')
    parser.add_argument('--device', type=str, default='cuda:0')

    config = parser.parse_args()
    config.Target_Dataset = ['RIM_ONE_r3', 'REFUGE', 'Drishti_GS', 'ORIGA', 'ACRIMA']
    config.Target_Dataset.remove(config.Source_Dataset)

    TTA = VPTTA(config)
    TTA.run()
