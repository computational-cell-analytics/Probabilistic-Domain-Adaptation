import os
import argparse
from glob import glob

import torch

import torch_em
from torch_em.model import UNet2d

from prob_utils.my_predictions import unet_prediction
from prob_utils.my_evaluations import run_lung_dice_evaluation
from prob_utils.my_datasets import get_jsrt_s1_loader, get_jsrt_s2_loader, get_montgomery_loader, get_nih_loader


def get_lung_loaders(lung_domain_name: str, root_input_dir: str):
    if lung_domain_name == "jsrt1":
        train_loader = get_jsrt_s1_loader(
            data_path=root_input_dir + "jsrt1/",
            split="train",
            batch_size=2
        )
        val_loader = get_jsrt_s1_loader(
            data_path=root_input_dir + "jsrt1/",
            split="val",
            batch_size=1
        )

    elif lung_domain_name == "jsrt2":
        train_loader = get_jsrt_s2_loader(
            data_path=root_input_dir + "jsrt2/",
            split="train",
            batch_size=2
        )
        val_loader = get_jsrt_s2_loader(
            data_path=root_input_dir + "jsrt2/",
            split="val",
            batch_size=1
        )

    elif lung_domain_name == "nih":
        train_loader = get_nih_loader(
            data_path=root_input_dir + "nih_processed/",
            split="train",
            batch_size=2
        )
        val_loader = get_nih_loader(
            data_path=root_input_dir + "nih_processed/",
            split="val",
            batch_size=1
        )

    elif lung_domain_name == "montgomery":
        train_loader = get_montgomery_loader(
            data_path=root_input_dir + "montgomery_processed/",
            split="train",
            batch_size=2
        )
        val_loader = get_montgomery_loader(
            data_path=root_input_dir + "montgomery_processed/",
            split="val",
            batch_size=1
        )

    return train_loader, val_loader


def do_unet_training(data_path, device):
    lung_domain_list = ["jsrt1", "jsrt2", "nih", "montgomery"]

    for lung_domain_name in lung_domain_list:
        model = UNet2d(
            in_channels=1,
            out_channels=1,
            final_activation="Sigmoid",
            depth=4,
            initial_features=64
        )
        model.to(device)

        train_loader, val_loader = get_lung_loaders(lung_domain_name=lung_domain_name, root_input_dir=data_path)

        source_trainer = torch_em.default_segmentation_trainer(
            name=f"unet-source-lung-{lung_domain_name}",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=1.0e-4,
            log_image_interval=1000
        )

        n_iterations = 100000
        source_trainer.fit(n_iterations)


def do_unet_predictions(device, data_path: str, pred_path: str):
    lung_domain_names = ["jsrt1", "jsrt2", "nih", "montgomery"]

    for source_lung_domain in lung_domain_names:
        model = UNet2d(
            in_channels=1,
            out_channels=1,
            final_activation="Sigmoid",
            depth=4,
            initial_features=64
        )

        model_save_dir = f"checkpoints/unet-source-lung-{source_lung_domain}/best.pt"

        if os.path.exists(model_save_dir) is False:
            print("The source model couldn't be found/hasn't been trained yet")
            continue

        model_state = torch.load(model_save_dir, map_location=torch.device('cpu'))["model_state"]
        model.load_state_dict(model_state)
        model.to(device)

        for target_lung_domain in lung_domain_names:
            output_path = pred_path + f"unet_source/source-{source_lung_domain}-target-{target_lung_domain}/"

            if target_lung_domain == "jsrt1":
                input_path = data_path + "jsrt1/test/org/*"
            elif target_lung_domain == "jsrt2":
                input_path = data_path + "jsr2/org_test/*"
            elif target_lung_domain == "nih":
                input_path = data_path + "nih_processed/images/test/*"
            elif target_lung_domain == "montgomery":
                input_path = data_path + "montgomery_processed/images/test/*"

            unet_prediction(input_path=glob(input_path), output_path=output_path, model=model, device=device)


def do_unet_evaluations(data_path: str, pred_path: str):
    lung_domain_names = ["jsrt1", "jsrt2", "nih", "montgomery"]

    for source_lung_domain in lung_domain_names:
        for target_lung_domain in lung_domain_names:
            unet_pred_path = pred_path + f"unet_source/source-{source_lung_domain}-target-{target_lung_domain}/"

            if os.path.exists(unet_pred_path) is False:
                print("The source model predictions couldn't be found/haven't been generated")
                continue

            if target_lung_domain == "jsrt1":
                gt_path = data_path + "jsrt1/test/label/"
            elif target_lung_domain == "jsrt2":
                gt_path = data_path + "jsrt2/label_test/"
            elif target_lung_domain == "nih":
                gt_path = data_path + "nih_processed/labels/test/"
            elif target_lung_domain == "montgomery":
                gt_path = data_path + "montgomery_processed/labels/test/"

            run_lung_dice_evaluation(gt_path, unet_pred_path, lung_domain=target_lung_domain)

            print(f"Dice on {target_lung_domain} from {source_lung_domain}")


def main(args):
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        print("Training 2D UNet on Lung X-Ray datasets")
        do_unet_training(data_path=args.data, device=device)

    if args.predict:
        print("Getting predictions on Lung X-Ray datasets from the trained UNet")
        do_unet_predictions(data_path=args.data, pred_path=args.pred_path, device=device)

    if args.evaluate:
        print("Evaluating the UNet predictions of Lung X-Ray datasets")
        do_unet_evaluations(data_path=args.data, pred_path=args.pred_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action='store_true', help="Enables UNet training on Lung X-Ray datasets")
    parser.add_argument("--predict", action='store_true', help="Obtains UNet predictions on Lung X-Ray test-sets")
    parser.add_argument("--evaluate", action='store_true', help="Evaluates UNet predictions")

    parser.add_argument("--data", type=str, help="Path where the dataset already exists for Lung X-Rays")
    parser.add_argument(
        "--pred_path", type=str, default="~/predictions/lung-xray/", help="Path where predictions will be saved"
    )

    args = parser.parse_args()
    main(args)
