import os
from glob import glob

import numpy as np
import imageio.v3 as imageio

from prob_utils.my_utils import dice_score


def run_dice_evaluation(gt_f_path, pred_path, subtype=None):
    """Dice evaluation
    """
    gt_dir = glob(gt_f_path)

    my_dice_list = []
    for my_path in gt_dir:
        fname = os.path.basename(my_path)

        if subtype == "lucchi":
            fname = f"mask{int(fname[:-4]):04}.tif"
        elif subtype == "urocell":
            fname = fname.replace("_gt", "_image")

        f_pred_path = os.path.join(pred_path, f"{fname[:-4]}.tif")

        my_pred = imageio.imread(f_pred_path)
        gt = imageio.imread(my_path)
        gt = (gt > 0).astype("uint8")

        if subtype == "lucchi":
            gt = gt[:, :, 0] if gt.ndim > 2 else gt

        my_dice = dice_score(my_pred, gt, threshold_seg=0.5)
        my_dice_list.append(my_dice)

    print(f"Average Dice Score for '{subtype}' - {round(sum(my_dice_list) / len(my_dice_list), 3)}")


def run_lung_dice_evaluation(gt_f_path, pred_path, lung_domain):
    'Dice evaluation for Lung Dataset'

    gt_path = gt_f_path + "*"
    gt_dir = glob(gt_path)

    my_dice_list = []

    for my_path in gt_dir:
        imagename = my_path.split('/')[-1]
        f_pred_path = pred_path + imagename[:-4] + ".tif"

        if lung_domain == "jsrt2":
            f_pred_path = pred_path + imagename[:-10] + ".tif"

        my_pred = imageio.imread(f_pred_path)
        gt = imageio.imread(my_path)
        gt = np.where(gt != 0, 1, gt)

        my_dice = dice_score(my_pred, gt, threshold_gt=0)
        my_dice_list.append(my_dice)

    print(f"Average Dice Score - {round(sum(my_dice_list)/len(my_dice_list), 3)}")


def run_em_dice_evaluation(gt_f_path, pred_path, model):
    'Dice evaluation for EM datasets'

    gt_path = gt_f_path + "*"
    gt_dir = glob(gt_path)

    my_dice_list = []

    for my_path in gt_dir:

        gt = imageio.imread(my_path)
        gt = np.where(gt != 0, 1, gt)

        imagename = my_path.split('/')[-1]
        f_pred_path = pred_path + imagename

        if model == "vnc":
            f_pred_path = pred_path + imagename[:-4] + ".tif"
        elif model == "lucchi":
            f_pred_path = pred_path + f"mask{int(imagename[:-4]):04}.tif"
            gt = gt[:, :, 0] if len(gt.shape) > 2 else gt
        elif model == 'mitoem':
            f_pred_path = pred_path + "im" + imagename[3:]

        my_pred = imageio.imread(f_pred_path)

        my_dice = dice_score(my_pred, gt, threshold_gt=0)
        my_dice_list.append(my_dice)

    print(f"Average Dice Score - {round(sum(my_dice_list)/len(my_dice_list), 3)}")


def run_dice_evaluation_for_pseudo(gt_f_path, pred_path, consensus_mask_path, model='punet'):
    'Dice evaluation for LiveCELL Pseudo Labels with Consensus Responses'

    gt_path = gt_f_path + "*.tif"
    gt_dir = glob(gt_path)

    my_list = []
    for my_path in gt_dir:
        imagename = my_path.split('/')[-1]
        f_pred_path = pred_path + imagename
        cm_path = consensus_mask_path + imagename

        if model == 'unet':
            f_pred_path = pred_path + imagename[:-4] + "-c0.tif"

        my_pred = imageio.imread(f_pred_path)
        gt = imageio.imread(my_path)
        consensus_mask = imageio.imread(cm_path)
        gt = np.where(gt != 0, 1, gt)
        consensus_mask = np.where(consensus_mask == 1, True, False)  # to get boolean values
        _my_pred = my_pred[consensus_mask]
        _gt = gt[consensus_mask]

        my_dice = dice_score(_my_pred, _gt, threshold_gt=0)

        my_list.append(my_dice)

    print(f"Average Dice over all {model} Predictions is - {round(sum(my_list)/len(my_list), 3)}")
