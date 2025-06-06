import os
import argparse

import torch

from torch_em.loss import DiceLoss

from prob_utils.my_models import ProbabilisticUnet
from prob_utils.my_trainer import PseudoTrainerPUNet
from prob_utils.my_datasets import get_my_livecell_loader
from prob_utils.my_evaluations import run_dice_evaluation
from prob_utils.my_predictions import punet_prediction, punet_pseudo_prediction


def do_punet_source_predictions(device, data_path: str, pred_path: str):
    cell_list = ['A172', 'BT474', 'BV2', 'Huh7', 'MCF7', 'SHSY5Y', 'SkBr3', 'SKOV3']

    model = ProbabilisticUnet(
        input_channels=1,
        num_classes=1,
        num_filters=[64, 128, 256, 512],
        latent_dim=6,
        no_convs_fcomb=3,
        beta=1.0,
        rl_swap=True
    )

    for cellname in cell_list:
        model_save_dir = f"checkpoints/punet-source-livecell-{cellname}/best.pt"

        if os.path.exists(model_save_dir) is False:
            print("The source trained model couldn't be found/hasn't been trained yet")
            continue

        model_state = torch.load(model_save_dir, map_location=torch.device('cpu'))["model_state"]
        model.load_state_dict(model_state)
        model.to(device)

        for cellname_ in cell_list:
            split_name = "livecell_train_val_images"

            input_path = data_path + f"images/{split_name}/"
            output_path = pred_path + f"punet_source_predictions/{cellname}/"

            punet_pseudo_prediction(
                input_image_path=input_path,
                output_pred_path=output_path,
                model=model,
                device=device,
                prior_samples=16,
                cellname_=cellname_,
                split_name=split_name
            )


def do_punet_target_training(args, device, data_path: str, pred_path: str, patch_shape=(256, 256)):
    cell_types = ['A172', 'BT474', 'BV2', 'Huh7', 'MCF7', 'SHSY5Y', 'SkBr3', 'SKOV3']

    for trg in cell_types:
        for src in cell_types:
            if src != trg:
                print(f"Transferring {src} learnings on {trg}")

                pseudo_label_path = pred_path + f"punet_source_predictions/{src}/"

                target_train_loader = get_my_livecell_loader(
                    path=data_path,
                    split='train',
                    patch_shape=patch_shape,
                    batch_size=2,
                    cell_types=[trg],
                    label_path=pseudo_label_path,
                    label_dtype=torch.float32
                )

                target_val_loader = get_my_livecell_loader(
                    path=data_path,
                    split='val',
                    patch_shape=patch_shape,
                    batch_size=1,
                    cell_types=[trg],
                    label_path=pseudo_label_path,
                    label_dtype=torch.float32
                )

                model = ProbabilisticUnet(
                    input_channels=1,
                    num_classes=1,
                    num_filters=[64, 128, 256, 512],
                    latent_dim=6,
                    no_convs_fcomb=3,
                    beta=1.0,
                    rl_swap=True,
                    consensus_masking=args.consensus
                )

                optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.9, patience=10, verbose=True
                )
                model.to(device)

                my_name = f"punet-livecell-source-{src}-target-{trg}"

                target_trainer = PseudoTrainerPUNet(
                    name=my_name if not args.consensus else my_name + "-consensus",
                    model=model,
                    train_loader=target_train_loader,
                    val_loader=target_val_loader,
                    loss=DiceLoss(),
                    metric=DiceLoss(),
                    logger=None,
                    device=device,
                    lr_scheduler=scheduler,
                    optimizer=optimizer,
                    log_image_interval=1000,
                    mixed_precision=True
                )

                n_iterations = 100000
                target_trainer.fit(n_iterations)


def do_punet_target_predictions(args, device, data_path: str, pred_path: str):
    cell_list = ['A172', 'BT474', 'BV2', 'Huh7', 'MCF7', 'SHSY5Y', 'SkBr3', 'SKOV3']

    model = ProbabilisticUnet(
        input_channels=1,
        num_classes=1,
        num_filters=[64, 128, 256, 512],
        latent_dim=6,
        no_convs_fcomb=3,
        beta=1.0,
        rl_swap=True
    )

    for trg in cell_list:
        for src in cell_list:
            if src != trg:
                if args.consensus:
                    model_save_dir = f"checkpoints/punet-livecell-source-{src}-target-{trg}-consensus/best.pt"
                else:
                    model_save_dir = f"checkpoints/punet-livecell-source-{src}-target-{trg}/best.pt"

                model_state = torch.load(model_save_dir, map_location=torch.device('cpu'))["model_state"]
                model.load_state_dict(model_state)
                model.to(device)

                input_path = data_path + f"images/livecell_test_images/{trg}*.tif"
                output_path = pred_path + f"punet_target/source-{src}-target-{trg}/"

                punet_prediction(
                    input_image_path=input_path,
                    output_pred_path=output_path,
                    model=model,
                    device=device,
                    prior_samples=16
                )


def do_punet_target_evaluations(data_path: str, pred_path: str):
    cell_list = ['A172', 'BT474', 'BV2', 'Huh7', 'MCF7', 'SHSY5Y', 'SkBr3', 'SKOV3']

    for trg in cell_list:
        gt_path = data_path + f"annotations/livecell_test_images/{trg}/"
        for src in cell_list:
            if src != trg:
                punet_pred_path = pred_path + f"punet_target/source-{src}-target-{trg}/"

                if os.path.exists(punet_pred_path) is False:
                    print("The model predictions haven't been generated, hence no evaluation")
                    continue

                run_dice_evaluation(gt_path, punet_pred_path)
                print(f"dice for {trg} from {src}-{trg}")


def main():
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.get_pseudo_labels:
        print("Generating the Pseudo Labels and Consensus Masks for the LiveCELL dataset")
        do_punet_source_predictions(device=device, data_path=args.data, pred_path=args.pred_path)

    if args.train:
        print("Training PUNet using Pseudo Labels on LiveCELL dataset")
        do_punet_target_training(args, data_path=args.data, pred_path=args.pred_path, device=device)

    if args.predict:
        print("Getting predictions on LiveCELL dataset from the trained PUNet target-transfer framework")
        do_punet_target_predictions(args, data_path=args.data, pred_path=args.pred_path, device=device)

    if args.evaluate:
        print("Evaluating the target-transfer predictions of LiveCELL dataset")
        do_punet_target_evaluations(data_path=args.data, pred_path=args.pred_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--get_pseudo_labels", action='store_true',
        help="Generates the Pseudo Labels and Consensus Masks, subjected to the already trained source domains"
    )
    parser.add_argument("--train", action='store_true', help="Enables target-transfer training on LiveCELL dataset")
    parser.add_argument(
        "--predict", action='store_true', help="Obtains target-transfer predictions on LiveCELL test-set"
    )
    parser.add_argument("--evaluate", action='store_true', help="Evaluates target-transfer predictions")
    parser.add_argument("--consensus", action='store_true', help="Activates Consensus Masking in the network")

    parser.add_argument(
        "--data", type=str, default="~/data/livecell/",
        help="Path where the dataset already exists/will be downloaded by the dataloader"
        )
    parser.add_argument(
        "--pred_path", type=str, default="~/predictions/livecell/", help="Path where predictions will be saved"
    )

    args = parser.parse_args()
    main(args)
