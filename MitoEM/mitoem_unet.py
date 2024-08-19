import argparse

import torch

import torch_em
from torch_em.model import UNet2d
from torch_em.data import MinForegroundSampler
from torch_em.data.datasets import get_mitoem_loader

from prob_utils.my_predictions import unet_prediction
from prob_utils.my_evaluations import run_dice_evaluation


def get_mitoem_loaders(
    data_path,
    patch_shape=(1, 512, 512),
    my_sampler=MinForegroundSampler(min_fraction=0.05)
):
    train_loader = get_mitoem_loader(
        path=data_path,
        splits="train",
        patch_shape=patch_shape,
        batch_size=2,
        ndim=2,
        binary=True,
        sampler=my_sampler,
        download=True
    )

    val_loader = get_mitoem_loader(
        path=data_path,
        splits="val",
        patch_shape=patch_shape,
        batch_size=1,
        ndim=2,
        binary=True,
        sampler=my_sampler,
        download=True
    )

    return train_loader, val_loader


def do_unet_training(device, data_path: str):
    train_loader, val_loader = get_mitoem_loaders(data_path)

    model = UNet2d(
        in_channels=1,
        out_channels=1,
        final_activation="Sigmoid",
        depth=4,
        initial_features=64
    )
    model.to(device)

    trainer = torch_em.default_segmentation_trainer(
        name="unet-source-mitoemv2",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-5,
        log_image_interval=1000,
        mixed_precision=True
    )

    n_iterations = 100000
    trainer.fit(n_iterations)


def do_unet_predictions(device, data_path: str, pred_path: str, em_type: str):
    root_output = pred_path + "unet_predictions/"

    if em_type == "lucchi":
        input_path = data_path + "lucchi/Lucchi++/Test_In/*"
        output_path = root_output + "lucchi/"
    elif em_type == "vnc":
        input_path = data_path + "vnc/groundtruth-drosophila-vnc-master/stack1/raw/*"
        output_path = root_output + "vnc/"

    model = UNet2d(
        in_channels=1,
        out_channels=1,
        final_activation="Sigmoid",
        depth=4,
        initial_features=64
    )
    model_save_dir = "checkpoints/unet-source-mitoemv2/best.pt"
    model_state = torch.load(model_save_dir, map_location=torch.device('cpu'))["model_state"]
    model.load_state_dict(model_state)
    model.to(device)

    unet_prediction(input_path=input_path, output_path=output_path, model=model, device=device)


def do_unet_evaluations(data_path: str, pred_path: str, em_type: str):
    root_output = pred_path + "unet_predictions/"

    if em_type == "lucchi":
        gt_path = data_path + "lucchi/Lucchi++/Test_Out/"
        output_path = root_output + "lucchi"

    elif em_type == "vnc":
        gt_path = data_path + "vnc/groundtruth-drosophila-vnc-master/stack1/mitochondria/"
        output_path = root_output + "vnc/"

    run_dice_evaluation(gt_f_path=gt_path, pred_path=output_path, model=em_type)


def main(args):
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        print("Training a 2D UNet on MitoEM dataset")
        do_unet_training(data_path=args.data, device=device)

    if args.predict:
        print("Getting predictions on Lucchi/VNC datasets from UNet trained on MitoEM")
        do_unet_predictions(data_path=args.data, pred_path=args.pred_path, device=device)

    if args.evaluate:
        print("Evaluating the UNet predictions")
        do_unet_evaluations(data_path=args.data, pred_path=args.pred_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action='store_true', help="Enables UNet training on MitoEM dataset")
    parser.add_argument("--predict", action='store_true', help="Obtains UNet predictions on Lucchi/VNC datasets")
    parser.add_argument("--evaluate", action='store_true', help="Evaluates UNet predictions")

    parser.add_argument(
        "--data", type=str, default="~/data/",
        help="Path where the dataset already exists/will be downloaded by the dataloader"
    )
    parser.add_argument(
        "--pred_path", type=str, default="~/predictions/mitoem/", help="Path where predictions will be saved"
    )

    args = parser.parse_args()
    main(args)
