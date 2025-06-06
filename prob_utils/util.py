import numpy as np

import torch


class DummyLoss(torch.nn.Module):
    pass


def my_standardize_torch(tensor, mean=None, std=None, axis=None, eps=1e-7):
    mean = tensor.mean() if mean is None else mean
    tensor -= mean
    std = tensor.std() if std is None else std
    tensor /= (std + eps)
    return tensor


def dice_score(segmentation, groundtruth, threshold_seg=None, threshold_gt=None):
    """ Compute the dice score between binarized segmentation and ground-truth.
    Arguments:
        segmentation [np.ndarray] - candidate segmentation to evaluate
        groundtruth [np.ndarray] - groundtruth
        threshold_seg [float] - the threshold applied to the segmentation.
            If None the segmentation is not thresholded.
        threshold_gt [float] - the threshold applied to the ground-truth.
            If None the ground-truth is not thresholded.
    Returns:
        float - the dice score
    """
    assert segmentation.shape == groundtruth.shape, f"{segmentation.shape}, {groundtruth.shape}"
    if threshold_seg is None:
        seg = segmentation
    else:
        seg = segmentation > threshold_seg
    if threshold_gt is None:
        gt = groundtruth
    else:
        gt = groundtruth > threshold_gt

    nom = 2 * np.sum(gt * seg)
    denom = np.sum(gt) + np.sum(seg)

    eps = 1e-7
    score = float(nom) / float(denom + eps)
    return score
