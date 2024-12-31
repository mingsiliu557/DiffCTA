#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np
from medpy import metric
from sklearn.metrics import roc_auc_score
from .tta import abl,rsa
from skimage.segmentation import random_walker
from skimage import color
import torch
import pydensecrf.utils as utils
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
import torch.nn.functional as F



def assert_shape(test, reference):

    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
        test.shape, reference.shape)


class ConfusionMatrix:

    def __init__(self, test=None, reference=None):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):

        self.test = test
        self.reset()

    def set_reference(self, reference):

        self.reference = reference
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):

        if self.test is None or self.reference is None:
            raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

        assert_shape(self.test, self.reference)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full


def dice(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """2TP / (2TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(2. * tp / (2 * tp + fp + fn))


def jaccard(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fp + fn))


def precision(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fp))


def sensitivity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fn))


def recall(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FN)"""

    return sensitivity(test, reference, confusion_matrix, nan_for_nonexisting, **kwargs)


def specificity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tn / (tn + fp))


def accuracy(test=None, reference=None, confusion_matrix=None, **kwargs):
    """(TP + TN) / (TP + FP + FN + TN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return float((tp + tn) / (tp + fp + tn + fn))


def fscore(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, beta=1., **kwargs):
    """(1 + b^2) * TP / ((1 + b^2) * TP + b^2 * FN + FP)"""

    precision_ = precision(test, reference, confusion_matrix, nan_for_nonexisting)
    recall_ = recall(test, reference, confusion_matrix, nan_for_nonexisting)

    return (1 + beta*beta) * precision_ * recall_ /\
        ((beta*beta * precision_) + recall_)


def false_positive_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FP / (FP + TN)"""

    return 1 - specificity(test, reference, confusion_matrix, nan_for_nonexisting)


def false_omission_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FN / (TN + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(fn / (fn + tn))


def false_negative_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FN / (TP + FN)"""

    return 1 - sensitivity(test, reference, confusion_matrix, nan_for_nonexisting)


def true_negative_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FP)"""

    return specificity(test, reference, confusion_matrix, nan_for_nonexisting)


def false_discovery_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FP / (TP + FP)"""

    return 1 - precision(test, reference, confusion_matrix, nan_for_nonexisting)


def negative_predictive_value(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FN)"""

    return 1 - false_omission_rate(test, reference, confusion_matrix, nan_for_nonexisting)


def total_positives_test(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TP + FP"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tp + fp


def total_negatives_test(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TN + FN"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tn + fn


def total_positives_reference(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TP + FN"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tp + fn


def total_negatives_reference(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TN + FP"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tn + fp


def hausdorff_distance(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.hd(test, reference, voxel_spacing, connectivity)


def hausdorff_distance_95(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return 100. #float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.hd95(test, reference, voxel_spacing, connectivity)


def avg_surface_distance(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return 100.#float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.asd(test, reference, voxel_spacing, connectivity)


def avg_surface_distance_symmetric(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.assd(test, reference, voxel_spacing, connectivity)

def post_process(abl_f, rsa_f, pred):
    if abl_f:
        pred = abl(pred, for_which_classes=[1])
    if rsa_f:
        # pred = rsa(pred, for_which_classes=[1], minimum_valid_object_size={1: 120})
        pred = rsa(pred, for_which_classes=[1], minimum_valid_object_size={1: 10})
    return pred


def crf_post_process(image, pred):
    """
    Perform CRF post-processing on predicted logits.

    Args:
        image (torch.Tensor or np.ndarray): The normalized image tensor of shape (b, 3, h, w) with values in [0, 1].
        pred (torch.Tensor or np.ndarray): Predicted logits after sigmoid of shape (b, 2, h, w).
        abl_f (bool): Placeholder for additional processing flags (default: False).
        rsa_f (bool): Placeholder for additional processing flags (default: False).

    Returns:
        np.ndarray: CRF-processed binary masks of shape (b, h, w).
    """
    b, c, h, w = pred.shape

    if isinstance(image, np.ndarray):
        image_np = image
    else:
        image_np = image.detach().cpu().numpy()

    if isinstance(pred, np.ndarray):
        pred_np = pred
    else:
        pred_np = pred.detach().cpu().numpy()

    # Convert image to uint8 range for CRF (expected range [0, 255])
    image_np = (image_np * 255).astype(np.uint8)

    processed_preds = []
    for i in range(b):
        img = np.transpose(image_np[i], (1, 2, 0))  # Convert to HxWxC

        # Predicted probabilities (after sigmoid)
        probs = pred_np[i]

        # Create CRF model
        d = dcrf.DenseCRF2D(w, h, c)

        # Unary potentials: negative log of probabilities
        U = unary_from_softmax(probs)
        d.setUnaryEnergy(U)

        # Add pairwise terms
        feats_gaussian = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats_gaussian, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

        feats_bilateral = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13), img=img, chdim=2)
        d.addPairwiseEnergy(feats_bilateral, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

        # Run CRF inference
        Q = d.inference(5)

        # Get refined predictions with CRF
        refined_pred = np.array(Q).reshape((c, h, w))
        processed_preds.append(refined_pred)

    # Stack predictions to form output of shape (b, c, h, w)
    processed_preds = np.stack(processed_preds, axis=0)
    return processed_preds.astype(np.float32)


# def seg_entropy(probs):
#     entropy = -probs * torch.log(probs + 1e-6) - (1 - probs) * torch.log(1 - probs + 1e-6)

#     # 如果是多通道的图像（c > 1），可以按通道平均
#     entropy = entropy.mean(dim=1)  # 对通道维度求平均，shape: (b, h, w)

#     # 计算每个像素的置信度
#     confidence = probs  # 置信度就是概率值

#     # 如果需要平均置信度，可以对所有像素的置信度取平均
#     average_confidence = confidence.mean(dim=[1, 2, 3])  # shape: (b,)
    
#     return average_confidence

def select_ensemble(logit1, logit2):

    # 计算每个logit的熵，按通道计算
    entropy1 = -logit1 * torch.log(logit1 + 1e-6) - (1 - logit1) * torch.log(1 - logit1 + 1e-6)
    entropy2 = -logit2 * torch.log(logit2 + 1e-6) - (1 - logit2) * torch.log(1 - logit2 + 1e-6)

    # 不对熵进行通道维度的平均，而是保留每个类别的熵信息
    # 这里entropy1和entropy2的形状将保持为 (b, 2, h, w)，表示每个类别的熵
    # 不需要进行mean(dim=1)

    # 生成Mask，表示哪些位置满足 entropy1 < entropy2
    mask = entropy1 < entropy2  # shape: (b, 2, h, w), 布尔值

    # 执行集成：根据Mask选择logit1和logit2的集成方式
    # 在mask为True的位置，使用 (logit1 + logit2) / 2；在mask为False的位置，保留logit1
    logit_ensemble = torch.where(mask,  # 在mask为True的地方
                                (logit1 + logit2) / 2,  # 集成logit1和logit2
                                logit1)  # 否则保留logit1
    return logit_ensemble


def dynamic_weighted_mixing(logit, logit_g):
    """
    输入两个logit（shape: b, c, h, w）输出基于动态权重的混合logit。
    
    logit: Tensor of shape (b, c, h, w) - 来自源模型的logit
    logit_g: Tensor of shape (b, c, h, w) - 来自生成模型的logit
    
    返回: Tensor of shape (b, c, h, w) - 动态加权后的logit
    """
    # 计算每个像素的预测置信度，即logit的最大值（可以用其他度量来替代）
    confidence_source = torch.max(logit, dim=1, keepdim=True)[0]  # 获取每个像素的最大logit值
    confidence_gen = torch.max(logit_g, dim=1, keepdim=True)[0]  # 获取每个像素的最大logit值
    
    # 归一化置信度值，使得它们在0到1之间
    confidence_source = F.softmax(confidence_source, dim=1)  # 归一化每个像素的置信度
    confidence_gen = F.softmax(confidence_gen, dim=1)  # 归一化每个像素的置信度
    
    # 计算权重因子（假设可以根据置信度进行动态调整）
    weight_source = confidence_source / (confidence_source + confidence_gen)
    weight_gen = 1 - weight_source  # 对应的权重
    
    # 对logit进行加权平均
    integrated_logit = weight_source * logit + weight_gen * logit_g
    
    return integrated_logit


def data_process(pred, label, image, threshold=0.5, abl_f = False, rsa_f = False):
    '''
    pred = np.array(pred*255).astype(np.uint8)
    label = np.array(label*255).astype(np.uint8)

    for N in range(pred.shape[0]):
        for C in range(pred.shape[1]):
            _, pred[N][C] = cv2.threshold(pred[N][C], 0, 255, cv2.THRESH_OTSU)
            _, label[N][C] = cv2.threshold(label[N][C], 0, 255, cv2.THRESH_OTSU)
    pred, label = pred // 255, label // 255
    '''
    pred = np.array(pred)
    label = np.array(label)
    #pred = crf_post_process(image, pred)
    b, c, h, w = pred.shape

    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0

    
    for i in range(b):
        for j in range(c):
            pred[i, j] = post_process(abl_f, rsa_f, pred[i, j])

    return pred.astype(np.uint8), label.astype(np.uint8)


def dice_compute(test, reference):
    batch_size = reference.shape[0]
    disc_dices, cup_dices = [], []

    for batch in range(batch_size):
        disc_dice, cup_dice = dice(test=test[batch][0], reference=reference[batch][0]), \
                              dice(test=test[batch][1], reference=reference[batch][1])
        disc_dices.append(disc_dice)
        cup_dices.append(cup_dice)
    return disc_dices, cup_dices


def asd_compute(test, reference):
    batch_size = reference.shape[0]
    disc_asds, cup_asds = [], []

    for batch in range(batch_size):
        disc_asd, cup_asd = avg_surface_distance(test=test[batch][0], reference=reference[batch][0]), \
                            avg_surface_distance(test=test[batch][1], reference=reference[batch][1])
        disc_asds.append(disc_asd)
        cup_asds.append(cup_asd)
    return disc_asds, cup_asds


def hd_compute(test, reference):
    batch_size = reference.shape[0]
    disc_hds, cup_hds = [], []

    for batch in range(batch_size):
        disc_hd, cup_hd = hausdorff_distance_95(test=test[batch][0], reference=reference[batch][0]), \
                          hausdorff_distance_95(test=test[batch][1], reference=reference[batch][1])
        disc_hds.append(disc_hd)
        cup_hds.append(cup_hd)
    return disc_hds, cup_hds


def dice_metric(pred, label):
    batch_size = pred.shape[0]
    disc_dices, cup_dices = [], []
    smooth = 1e-6

    for batch in range(batch_size):
        disc_intersection = (pred[batch][0] * label[batch][0]).sum()
        disc_dice = (2 * disc_intersection + smooth) / (pred[batch][0].sum() + label[batch][0].sum() + smooth)
        cup_intersection = (pred[batch][-1] * label[batch][-1]).sum()
        cup_dice = (2 * cup_intersection + smooth) / (pred[batch][-1].sum() + label[batch][-1].sum() + smooth)

        disc_dices.append(disc_dice*100.)
        cup_dices.append(cup_dice*100.)

    return disc_dices, cup_dices



def calculate_metrics(test, reference, image, rsa_f = False, abl_f = False):
    test, reference = data_process(pred=test, label=reference, image=image, threshold=0.5, abl_f=abl_f, rsa_f=rsa_f)
    #test, reference = data_process_crf(pred=test, image=image,label=reference, threshold=0.5, abl_f=abl_f, rsa_f=rsa_f)
    disc_dice, cup_dice = dice_metric(test, reference)
    disc_dis, cup_dis = asd_compute(test, reference)
    # disc_dis, cup_dis = hd_compute(test, reference)
    return [disc_dice, disc_dis, cup_dice, cup_dis]


ALL_METRICS = {
    "False Positive Rate": false_positive_rate,
    "Dice": dice,
    "Jaccard": jaccard,
    "Hausdorff Distance": hausdorff_distance,
    "Hausdorff Distance 95": hausdorff_distance_95,
    "Precision": precision,
    "Recall": recall,
    "Avg. Symmetric Surface Distance": avg_surface_distance_symmetric,
    "Avg. Surface Distance": avg_surface_distance,
    "Accuracy": accuracy,
    "False Omission Rate": false_omission_rate,
    "Negative Predictive Value": negative_predictive_value,
    "False Negative Rate": false_negative_rate,
    "True Negative Rate": true_negative_rate,
    "False Discovery Rate": false_discovery_rate,
    "Total Positives Test": total_positives_test,
    "Total Negatives Test": total_negatives_test,
    "Total Positives Reference": total_positives_reference,
    "total Negatives Reference": total_negatives_reference
}