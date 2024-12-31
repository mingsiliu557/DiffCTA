"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np

import torch as th

import torchvision.transforms as transforms

from torch.nn import functional as F

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs




# # Existing AugMixWrapper class and marginal_entropy_loss function
class AugMixWrapper:
    def __init__(self, num_aug, severity=3, alpha=1.0):
        """
        AugMixWrapper is to get the mean augmentations.
        
        Args:
            severity (int): AugMix severity.
            alpha (float): Dirichlet to settle down the weight of every augmentation.
        """
        self.severity = severity
        self.num_aug = num_aug
        self.alpha = alpha
        self.augmentations = transforms.AugMix(severity=self.severity, mixture_width=self.num_aug)

    def __call__(self, x):
        # Apply augmentations and get the mean augmentation
        original_x = x.clone()  # Copy the original tensor to keep the computation graph intact
        if x.dtype != th.uint8:
            x = (x * 255).clamp(0, 255).to(th.uint8)  # Convert to uint8 if necessary
        aug_img = self.augmentations(x)
        original_x.copy_(aug_img)  # Update the original tensor with the new augmented values
        return original_x


def marginal_entropy_loss(model, x):


    
    # Calculate marginal prediction
    marginal_pred = model(x)
    
    marginal_prob = th.softmax(marginal_pred, dim=-1)
    # Calculate marginal entropy
    marginal_entropy = -th.sum(marginal_prob * th.log(marginal_prob + 1e-6)) / marginal_prob.size(0)
    
    return marginal_entropy

def d_clip_loss(x, y, use_cosine=False):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    if use_cosine:
        distance = 1 - (x @ y.t()).squeeze()
    else:
        distance = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

    return distance




def clip_global_loss(self,x_in,text_embed):
    clip_loss = th.tensor(0)
    augmented_input = self.image_augmentations(x_in,num_patch=self.args.n_patch).add(1).div(2)
    clip_in = self.clip_normalize(augmented_input)
    image_embeds = self.clip_model.encode_image(clip_in).float()
    dists = d_clip_loss(image_embeds, text_embed)
    for i in range(self.args.batch_size):
        clip_loss = clip_loss + dists[i :: self.args.batch_size].mean()

    return clip_loss


def dice_loss(pred_mask, true_mask, epsilon=1e-6):
    """
    计算 Dice Loss
    参数:
    - pred_mask (Tensor): 预测的掩码，形状为 (batch_size, height, width) 或 (batch_size, num_classes, height, width)
    - true_mask (Tensor): 真实的掩码，形状与 pred_mask 相同
    - epsilon (float): 防止除以零的小常数
    
    返回:
    - dice_loss (Tensor): 计算得到的 Dice Loss
    """
    # Flatten 输入张量
    pred_mask = pred_mask.view(-1)
    true_mask = true_mask.view(-1)

    # 计算交集 (A ∩ B) 和并集 (A ∪ B)
    intersection = th.sum(pred_mask * true_mask)
    union = th.sum(pred_mask) + th.sum(true_mask)

    # 计算 Dice 系数，防止除以零，因此加上 epsilon
    dice_coefficient = (2.0 * intersection + epsilon) / (union + epsilon)

    # Dice Loss 是 1 减去 Dice 系数
    dice_loss = 1 - dice_coefficient

    return dice_loss