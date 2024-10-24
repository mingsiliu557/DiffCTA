"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np

import torch as th

import torchvision.transforms as transforms

import unittest

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


# Existing AugMixWrapper class and marginal_entropy_loss function
class AugMixWrapper:
    def __init__(self, severity=3, alpha=1.0):
        """
        AugMixWrapper is to get the mean augmentations.
        
        Args:
            severity (int): AugMix severity.
            alpha (float): Dirichlet to settle down the weight of every augmentation.
        """
        self.severity = severity
        self.alpha = alpha
        self.augmentations = transforms.AugMix(severity=self.severity)

    def __call__(self, x):
        # Apply augmentations and get the mean augmentation
        original_x = x.clone()  # Copy the original tensor to keep the computation graph intact
        if x.dtype != th.uint8:
            x = (x * 255).clamp(0, 255).to(th.uint8)  # Convert to uint8 if necessary
        
        weights = th.distributions.Dirichlet(th.tensor([self.alpha] * 3)).sample().to(x.device)
        mix = th.zeros_like(x, dtype=th.float32)
        
        for i in range(3):
            aug_img = self.augmentations(x)
            aug_img = aug_img.to(dtype=th.float32) / 255.0  # Ensure aug_img is float32 and normalized to [0, 1]
            mix += weights[i] * aug_img

        original_x.copy_(mix)  # Update the original tensor with the new augmented values
        return original_x

def augmentation(x, augmentations=5, severity=3, alpha=1.0):
    augmix = AugMixWrapper(severity=severity, alpha=alpha)
    augs = []
    for _ in range(augmentations):
        x_aug = augmix(x)
        augs.append(x_aug)

    # Calculate marginal prediction
    aug_final = sum(augs) / augmentations
    return aug_final

def marginal_entropy_loss(model, x, augmentations=5, severity=3, alpha=1.0):
    augmix = AugMixWrapper(severity=severity, alpha=alpha)

    aug_preds = []
    for _ in range(augmentations):
        x_aug = augmix(x)
        pred = model(x_aug)
        aug_preds.append(pred)
    
    # Calculate marginal prediction
    marginal_pred = sum(aug_preds) / augmentations
    
    # Calculate marginal entropy
    marginal_entropy = -th.sum(marginal_pred * th.log(marginal_pred + 1e-6)) / marginal_pred.size(0)
    
    return marginal_entropy

# Create a simple model for testing
class SimpleModel(th.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = th.nn.Linear(3 * 32 * 32, 10)  # Input is a 32x32 RGB image, output is 10 classes

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten to two dimensions
        return th.softmax(self.fc(x), dim=1)

# Unit test section
class TestAugMixWrapper(unittest.TestCase):
    def setUp(self):
        # Initialize model and input data
        self.model = SimpleModel()
        self.input_data = th.randn(1, 3, 32, 32)  # Randomly generate a 32x32 RGB image

    def test_augmix_wrapper(self):
        # Test the output shape of AugMixWrapper
        augmix = AugMixWrapper(severity=3, alpha=1.0)
        output = augmix(self.input_data)
        self.assertEqual(output.shape, self.input_data.shape, "Output shape should match input shape")

    def test_marginal_entropy_loss(self):
        # Test if the marginal entropy loss function runs correctly
        loss = marginal_entropy_loss(self.model, self.input_data, augmentations=5, severity=3, alpha=1.0)
        self.assertIsInstance(loss, th.Tensor, "Loss should be a tensor")
        self.assertGreaterEqual(loss.item(), 0, "Loss should be non-negative")
        print(f'the loss is {loss.item()}')

if __name__ == "__main__":
    unittest.main(exit=False)
