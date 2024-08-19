import argparse

import torch

from torch_em.data import MinForegroundSampler
from torch_em.data.datasets import get_mitoem_loader

from prob_utils.my_utils import DummyLoss
from prob_utils.my_models import ProbabilisticUnet
from prob_utils.my_predictions import punet_prediction
from prob_utils.my_trainer import PUNetTrainer, PUNetLogger
from prob_utils.my_evaluations import run_em_dice_evaluation


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


def do_punet_training(device, data_path: str):
    train_loader, val_loader = get_mitoem_loaders(path=data_path)

    model = ProbabilisticUnet(
        input_channels=1,
        num_classes=1,
        num_filters=[64, 128, 256, 512],
        latent_dim=6,
        no_convs_fcomb=3,
        beta=1.0,
        rl_swap=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=True)
    model = model.to(device)

    trainer = PUNetTrainer(
        name="punet-source-mitoemv2",
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        loss=DummyLoss(),
        metric=DummyLoss(),
        device=device,
        lr_scheduler=scheduler,
        logger=PUNetLogger,
        mixed_precision=True,
        log_image_interval=1000
    )

    n_iterations = 100000
    trainer.fit(n_iterations)


def do_punet_predictions(device, data_path: str, pred_path: str):
    em_types = ["vnc", "lucchi"]

    for em_data in em_types:
        root_output = pred_path + "punet_predictions/"
        output_path = root_output + f"{em_data}/"

        if em_data == "lucchi":
            input_path = data_path + "lucchi/Lucchi++/Test_In/*"
        elif em_data == "vnc":
            input_path = data_path + "vnc/groundtruth-drosophila-vnc-master/stack1/raw/*"

        model = ProbabilisticUnet(
            input_channels=1,
            num_classes=1,
            num_filters=[64, 128, 256, 512],
            latent_dim=6,
            no_convs_fcomb=3,
            beta=1.0,
            rl_swap=True
        )
        model_save_dir = "checkpoints/punet-source-mitoemv2/best.pt"
        model_state = torch.load(model_save_dir, map_location=torch.device('cpu'))["model_state"]
        model.load_state_dict(model_state)
        model.to(device)

        punet_prediction(input_image_path=input_path, output_pred_path=output_path, model=model, device=device)


def do_punet_evaluations(data_path: str, pred_path: str):
    em_types = ["vnc", "lucchi"]

    for em_data in em_types:
        root_output = pred_path + "punet_predictions/"
        output_path = root_output + f"{em_data}/"

        if em_data == "lucchi":
            gt_path = data_path + "lucchi/Lucchi++/Test_Out/"
        elif em_data == "vnc":
            gt_path = data_path + "vnc/groundtruth-drosophila-vnc-master/stack1/mitochondria/"

        run_em_dice_evaluation(gt_f_path=gt_path, pred_path=output_path, model=em_data)


def main(args):
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        print("Training PUNet on MitoEM dataset")
        do_punet_training(data_path=args.data, device=device)

    if args.predict:
        print("Getting predictions on Lucchi/VNC datasets from PUNet trained on MitoEM")
        do_punet_predictions(data_path=args.data, pred_path=args.pred_path, device=device)

    if args.evaluate:
        print("Evaluating the PUNet predictions")
        do_punet_evaluations(data_path=args.data, pred_path=args.pred_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action='store_true', help="Enables PUNet training on MitoEM dataset")
    parser.add_argument("--predict", action='store_true', help="Obtains PUNet predictions on Lucchi/VNC datasets")
    parser.add_argument("--evaluate", action='store_true', help="Evaluates PUNet predictions")

    parser.add_argument(
        "--data", type=str, default="~/data/",
        help="Path where the dataset already exists/will be downloaded by the dataloader"
    )
    parser.add_argument(
        "--pred_path", type=str, default="~/predictions/mitoem/", help="Path where predictions will be saved"
    )

    args = parser.parse_args()
    main(args)
