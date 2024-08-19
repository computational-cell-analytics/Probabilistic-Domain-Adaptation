import os
from glob import glob

import numpy as np
import imageio.v3 as imageio

import torch

import torch_em

from prob_utils.my_models import clean_folder


def punet_prediction(
    input_image_path,
    output_pred_path,
    model,
    prior_samples=8,
    device='cpu',
    mysig=torch.nn.Sigmoid()
):
    'function that generates predictions from the PUNet'
    os.makedirs(output_pred_path, exist_ok=True)
    clean_folder(output_pred_path)

    model.eval()
    with torch.no_grad():
        my_data_dir = input_image_path

        for i in glob(my_data_dir):

            my_image_name = i.split('/')[-1]

            my_patch = imageio.imread(i)
            my_patch = torch_em.transform.raw.standardize(my_patch)
            my_patch = torch.from_numpy(my_patch)
            my_patch = my_patch.unsqueeze(0).unsqueeze(0).to(device)

            model.forward(my_patch, None, training=False)

            samples_per_patch = [mysig(model.sample(testing=True)) for _ in range(prior_samples)]
            mypred = torch.stack(samples_per_patch, dim=0).sum(dim=0)/prior_samples
            mypred = mypred.detach().cpu().numpy().squeeze()

            pred_image_name = f"{my_image_name[:-4]}.tif"
            my_rand_name = output_pred_path + pred_image_name
            imageio.imwrite(my_rand_name, mypred)
            print(f"{my_image_name} prediction saved")


def punet_pseudo_prediction(
    input_image_path,
    output_pred_path,
    model,
    prior_samples=8,
    device='cpu',
    cellname_=None,
    split_name: str = None
):
    """Function to use trained punet on test samples (now for pseudo labelling)
    output_pred_path : Path where predictions will be saved
    input_image_path : Path where the input images are there
    model : weights initialised to the model architecture
    device : cpu or gpu
    prior_samples : set the number of times we sample from prior net
    """

    os.makedirs(output_pred_path, exist_ok=True)
    clean_folder(output_pred_path)

    # always define variables instead of using the magic values below in the code
    # this makes the code more readable and enables passing these as parameters later
    upper_threshold = 0.9
    lower_threshold = 0.1

    model.eval()
    with torch.no_grad():
        my_data_dir = input_image_path + f"{cellname_}*.tif"

        for i in glob(my_data_dir):
            my_image_name = i.split('/')[-1]
            my_patch = imageio.imread(i)
            my_patch = torch_em.transform.raw.standardize(my_patch)
            my_patch = torch.from_numpy(my_patch)
            my_patch = my_patch.unsqueeze(0).unsqueeze(0).to(device)
            model.forward(my_patch, None, training=False)

            samples_per_patch = []  # original samples b/w range [0,1]
            masks_per_patch = []  # The "confidence mask" (pixels that are <0.1, >0.9 for the pixels in all samples)

            for _ in range(prior_samples):
                mysig = torch.nn.Sigmoid()
                myval = model.sample(testing=True)
                myval = mysig(myval)
                samples_per_patch.append(myval)

            # average of all predicted range of p-values per n-samples
            mypred = torch.stack(samples_per_patch, dim=0).sum(dim=0)/prior_samples
            mypred = mypred.detach().cpu().numpy().squeeze()

            for sample in samples_per_patch:
                sample = sample.detach().cpu().numpy().squeeze()
                mask_sample = (sample >= upper_threshold) + (sample <= lower_threshold)
                masks_per_patch.append(mask_sample)

            # consensus mask
            consensus_mask = np.stack(masks_per_patch, axis=0).sum(axis=0)/prior_samples
            consensus_mask = np.where(consensus_mask == 1, 1, 0)

            dir1 = output_pred_path + f"annotations/{split_name}/{cellname_}/"
            dir2 = output_pred_path + f"consensus/{split_name}/{cellname_}/"
            os.makedirs(dir1, exist_ok=True)
            os.makedirs(dir2, exist_ok=True)

            name1 = dir1 + f'{my_image_name}'
            name2 = dir2 + f'{my_image_name}'

            imageio.imwrite(name1, mypred)
            imageio.imwrite(name2, consensus_mask.astype("uint8"))
            print(f"{my_image_name}'s predictions saved")
