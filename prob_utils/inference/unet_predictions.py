import os
import imageio.v2 as imageio

import torch
import torch_em
from torch_em.util.prediction import predict_with_padding

def unet_prediction(input_path:str, output_path:str, model, device):
    'function that generates UNet predictions'

    os.makedirs(output_path, exist_ok=True)
    with torch.no_grad():
        for img_dir in input_path:
            img_name = img_dir.split('/')[-1]
            input_img = imageio.imread(img_dir)

            input_img = torch_em.transform.raw.standardize(input_img)
            pred = predict_with_padding(model, input_img, 
                                        min_divisible=(16,16), device=device)
            pred = pred.squeeze()
            output_dir = output_path + f"{img_name[:-4]}.tif"
            imageio.imwrite(output_dir, pred)
            print(f"Saved image at {output_dir}")