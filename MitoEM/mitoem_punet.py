import os
import argparse

import torch

from prob_utils.my_utils import DummyLoss
from prob_utils.my_models import ProbabilisticUnet
from prob_utils.my_predictions import punet_prediction
from prob_utils.my_evaluations import run_dice_evaluation
from prob_utils.my_trainer import PUNetTrainer, PUNetLogger

from common import get_mitoem_loaders


def do_punet_training(device, data_path: str, save_root: str):
    train_loader, val_loader = get_mitoem_loaders(data_path)

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10)
    model = model.to(device)

    trainer = PUNetTrainer(
        name="punet-source-mitoem",
        save_root=save_root,
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
        compile_model=False,
        log_image_interval=1000
    )

    n_iterations = 100000
    trainer.fit(n_iterations)


def do_punet_predictions(device, data_path: str, pred_path: str, em_type: str, save_root: str):
    root_output = os.path.join(pred_path, "punet_predictions")
    output_path = os.path.join(root_output, em_type)

    if em_type == "lucchi":
        input_path = os.path.join(data_path, "lucchi", "Lucchi++", "Test_In", "*")
    elif em_type == "vnc":
        input_path = os.path.join(data_path, "vnc", "groundtruth-drosophila-vnc-master", "stack1", "raw", "*")
    elif em_type == "urocell":
        input_path = os.path.join(data_path, "urocell", "preprocessed", "*_image.tif")

    model = ProbabilisticUnet(
        input_channels=1,
        num_classes=1,
        num_filters=[64, 128, 256, 512],
        latent_dim=6,
        no_convs_fcomb=3,
        beta=1.0,
        rl_swap=True
    )
    model_save_dir = os.path.join(
        ("./" if save_root is None else save_root), "checkpoints", "punet-source-mitoem", "best.pt"
    )
    model_state = torch.load(model_save_dir, map_location="cpu", weights_only=False)["model_state"]
    model.load_state_dict(model_state)
    model.to(device)

    punet_prediction(input_image_path=input_path, output_pred_path=output_path, model=model, device=device)


def do_punet_evaluations(data_path: str, pred_path: str, em_type: str):
    root_output = os.path.join(pred_path, "punet_predictions")
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
        print("Training a 2D PUNet on MitoEM dataset")
        do_punet_training(data_path=os.path.join(args.data, "mitoem"), device=device, save_root=args.save_root)

    if args.predict:
        print("Getting predictions on Lucchi / VNC / UroCell datasets from PUNet trained on MitoEM")
        for em_type in ["vnc", "lucchi", "urocell"]:
            do_punet_predictions(
                data_path=args.data, pred_path=args.pred_path, device=device, em_type=em_type, save_root=args.save_root,
            )

    if args.evaluate:
        print("Evaluating the PUNet predictions")
        for em_type in ["vnc", "lucchi", "urocell"]:
            do_punet_evaluations(data_path=args.data, pred_path=args.pred_path, em_type=em_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', help="Enables PUNet training on MitoEM dataset")
    parser.add_argument("--predict", action='store_true', help="Obtains PUNet predictions on Lucchi/VNC datasets")
    parser.add_argument("--evaluate", action='store_true', help="Evaluates PUNet predictions")
    parser.add_argument(
        "--data", type=str, default="/mnt/lustre-grete/usr/u16934/data",
        help="Path where the dataset already exists/will be downloaded by the dataloader"
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
