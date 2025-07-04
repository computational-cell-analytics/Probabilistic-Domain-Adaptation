import os
import argparse

import torch

import torch_em
from torch_em.model import UNet2d

from prob_utils.my_predictions import unet_prediction
from prob_utils.my_evaluations import run_dice_evaluation

from common import get_mitoem_loaders


def do_unet_training(device, data_path: str, save_root: str):
    train_loader, val_loader = get_mitoem_loaders(data_path)

    model = UNet2d(
        in_channels=1,
        out_channels=1,
        final_activation="Sigmoid",
        depth=4,
        initial_features=64,
    )
    model.to(device)

    trainer = torch_em.default_segmentation_trainer(
        name="unet-source-mitoem",
        save_root=save_root,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-5,
        log_image_interval=1000,
        mixed_precision=True,
        compile_model=False,
    )

    n_iterations = 100000
    trainer.fit(n_iterations)


def do_unet_predictions(device, data_path: str, pred_path: str, em_type: str, save_root: str):
    root_output = os.path.join(pred_path, "unet_predictions")
    output_path = os.path.join(root_output, em_type)

    if em_type == "lucchi":
        input_path = os.path.join(data_path, "lucchi", "Lucchi++", "Test_In", "*")
    elif em_type == "vnc":
        input_path = os.path.join(data_path, "vnc", "groundtruth-drosophila-vnc-master", "stack1", "raw", "*")
    elif em_type == "urocell":
        input_path = os.path.join(data_path, "urocell", "preprocessed", "*_image.tif")

    model = UNet2d(
        in_channels=1,
        out_channels=1,
        final_activation="Sigmoid",
        depth=4,
        initial_features=64
    )

    model_save_dir = os.path.join(
        ("./" if save_root is None else save_root), "checkpoints", "unet-source-mitoem", "best.pt"
    )
    model_state = torch.load(model_save_dir, map_location="cpu", weights_only=False)["model_state"]
    model.load_state_dict(model_state)
    model.to(device)

    unet_prediction(input_path=input_path, output_path=output_path, model=model, device=device)


def do_unet_evaluations(data_path: str, pred_path: str, em_type: str):
    root_output = os.path.join(pred_path, "unet_predictions")
    output_path = os.path.join(root_output, em_type)

    if em_type == "lucchi":
        gt_path = os.path.join(data_path, "lucchi", "Lucchi++", "Test_Out", "*")
    elif em_type == "vnc":
        gt_path = os.path.join(data_path, "vnc", "groundtruth-drosophila-vnc-master", "stack1", "mitochondria", "*")
    elif em_type == "urocell":
        gt_path = os.path.join(data_path, "urocell", "preprocessed", "*_gt.tif")

    run_dice_evaluation(gt_f_path=gt_path, pred_path=output_path, subtype=em_type)


def main(args):
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        print("Training a 2D UNet on MitoEM dataset")
        do_unet_training(data_path=os.path.join(args.data, "mitoem"), device=device, save_root=args.save_root)

    if args.predict:
        print("Getting predictions on Lucchi / VNC / UroCell datasets from UNet trained on MitoEM")
        for em_type in ["vnc", "lucchi", "urocell"]:
            do_unet_predictions(
                data_path=args.data, pred_path=args.pred_path, device=device, em_type=em_type, save_root=args.save_root,
            )

    if args.evaluate:
        print("Evaluating the UNet predictions")
        for em_type in ["vnc", "lucchi", "urocell"]:
            do_unet_evaluations(data_path=args.data, pred_path=args.pred_path, em_type=em_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', help="Enables UNet training on MitoEM dataset")
    parser.add_argument("--predict", action='store_true', help="Obtains UNet predictions on Lucchi / VNC datasets")
    parser.add_argument("--evaluate", action='store_true', help="Evaluates UNet predictions")

    parser.add_argument(
        "--data", type=str, default="/mnt/lustre-grete/usr/u16934/data",
        help="Path where the dataset already exists/will be downloaded by the dataloader."
    )
    parser.add_argument(
        "--pred_path", type=str, default="/mnt/lustre-grete/usr/u16934/experiments/pda/source-mitoem",
        help="Path where predictions will be saved."
    )
    parser.add_argument(
        "--save_root", default=None, type=str, help="Path where the trained models are stored."
    )
    args = parser.parse_args()
    main(args)
