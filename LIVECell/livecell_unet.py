import os
import argparse
from glob import glob

import torch

import torch_em
from torch_em.model import UNet2d
from torch_em.data.datasets import get_livecell_loader

from prob_utils.my_predictions import unet_prediction
from prob_utils.my_evaluations import run_dice_evaluation


def do_unet_training(data_path: str, device, patch_shape=(256, 256)):
    cell_types_list = ['A172', 'BT474', 'BV2', 'Huh7', 'MCF7', 'SHSY5Y', 'SkBr3', 'SKOV3']

    for ctype in cell_types_list:
        os.makedirs(data_path, exist_ok=True)

        target_train_loader = get_livecell_loader(
            path=data_path,
            split='train',
            binary=True,
            patch_shape=patch_shape,
            batch_size=4,
            cell_types=[ctype],
            download=True
        )
        target_val_loader = get_livecell_loader(
            path=data_path,
            split='val',
            binary=True,
            patch_shape=patch_shape,
            batch_size=1,
            cell_types=[ctype],
            download=True
        )

        model = UNet2d(
            in_channels=1,
            out_channels=1,
            final_activation="Sigmoid",
            depth=4,
            initial_features=64
        )
        model.to(device)

        target_trainer = torch_em.default_segmentation_trainer(
            name=f"unet-source-livecell-{ctype}",
            model=model,
            train_loader=target_train_loader,
            val_loader=target_val_loader,
            device=device,
            learning_rate=1.0e-4,
            log_image_interval=1000
        )

        n_iterations = 100000
        target_trainer.fit(n_iterations)


def do_unet_predictions(data_path: str, pred_path: str, device):
    cell_types_list = ['A172', 'BT474', 'BV2', 'Huh7', 'MCF7', 'SHSY5Y', 'SkBr3', 'SKOV3']

    for ctype1 in cell_types_list:  # source's trained weights coming from "ctype1"
        model = UNet2d(
            in_channels=1,
            out_channels=1,
            final_activation="Sigmoid",
            depth=4,
            initial_features=64
        )

        model_save_dir = f"checkpoints/unet-source-livecell-{ctype1}/best.pt"

        if os.path.exists(model_save_dir) is False:
            print("The source model couldn't be found/hasn't been trained yet")
            continue

        model_state = torch.load(model_save_dir, map_location=torch.device('cpu'))["model_state"]
        model.load_state_dict(model_state)
        model.to(device)

        for ctype2 in cell_types_list:  # target's transfer going to "ctype2"
            input_path = glob(data_path + f"images/livecell_test_images/{ctype2}*.tif")
            output_path = pred_path + f"unet_source/{ctype1}/{ctype2}/"

            unet_prediction(input_path=input_path, output_path=output_path, model=model, device=device)


def do_unet_evaluations(data_path: str, pred_path: str):
    cell_types_list = ['A172', 'BT474', 'BV2', 'Huh7', 'MCF7', 'SHSY5Y', 'SkBr3', 'SKOV3']

    for ctype1 in cell_types_list:
        gt_dir = data_path + f"annotations/livecell_test_images/{ctype1}/"
        for ctype2 in cell_types_list:
            pred_dir = pred_path + f"unet_source/{ctype2}/{ctype1}/"

            if os.path.exists(pred_dir) is False:
                print("The source model predictions couldn't be found/haven't been generated")
                continue

            run_dice_evaluation(gt_dir, pred_dir)

            print(f"Dice for Target Cells - {ctype1} from Source Cells - {ctype2}")


def main(args):
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        print("Training a 2D UNet on LiveCELL dataset")
        do_unet_training(data_path=args.data, device=device)

    if args.predict:
        print("Getting predictions on LiveCELL dataset from the trained UNet")
        do_unet_predictions(data_path=args.data, pred_path=args.pred_path, device=device)

    if args.evaluate:
        print("Evaluating the UNet predictions of LiveCELL dataset")
        do_unet_evaluations(data_path=args.data, pred_path=args.pred_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action='store_true', help="Enables UNet training on LiveCELL dataset")
    parser.add_argument("--predict", action='store_true', help="Obtains UNet predictions on LiveCELL test-set")
    parser.add_argument("--evaluate", action='store_true', help="Evaluates UNet predictions")

    parser.add_argument(
        "--data", type=str, default="~/data/livecell/",
        help="Path where the dataset already exists/will be downloaded by the dataloader"
    )
    parser.add_argument(
        "--pred_path", type=str, default="~/predictions/livecell/", help="Path where predictions will be saved"
    )

    args = parser.parse_args()
    main(args)
