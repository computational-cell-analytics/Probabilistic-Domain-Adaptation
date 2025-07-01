import os
from glob import glob

import imageio.v3 as imageio

import torch

from torch_em.transform.raw import standardize
from torch_em.util.prediction import predict_with_halo, predict_with_padding


@torch.no_grad()
def unet_prediction(input_path: str, output_path: str, model: torch.nn.Module, device: str):
    """Function that generates UNet predictions.
    """
    model.eval()
    os.makedirs(output_path, exist_ok=True)

    for img_path in glob(input_path):
        img_name = os.path.basename(img_path)
        input_img = imageio.imread(img_path)

        tiling = True
        if tiling:
            pred = predict_with_halo(
                input_=input_img,
                model=model,
                gpu_ids=[device],
                block_shape=(384, 384),
                halo=(64, 64),
            )
        else:
            pred = predict_with_padding(
                model=model,
                input_=standardize(input_img),
                min_divisible=(16, 16),
                device=device,
            )

        pred = pred.squeeze()
        pred_path = os.path.join(output_path, f"{img_name[:-4]}.tif")
        imageio.imwrite(pred_path, pred, compression="zlib")
        print(f"Saved image at '{pred_path}'")
