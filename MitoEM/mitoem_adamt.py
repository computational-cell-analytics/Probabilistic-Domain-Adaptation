import os
import argparse

import torch

from prob_utils.my_utils import DummyLoss
from prob_utils.my_models import ProbabilisticUnet
from prob_utils.my_predictions import punet_prediction
from prob_utils.my_evaluations import run_dice_evaluation
from prob_utils.my_trainer import AdaMTTrainer, AdaMTLogger

from common import get_dual_loaders, get_mitoem_loaders


def do_adamt_training(args, device, em_types: str, data_path: str):
    if em_types is None:
        em_types = ["vnc", "lucchi", "urocell"]
    else:
        em_types = [em_types]

    for em_data in em_types:
        print(f"Training on {em_data} using AdaMT scheme")

        source_train_loader, _ = get_mitoem_loaders(os.path.join(data_path, "mitoem"))
        target_train_loader, target_val_loader = get_dual_loaders(em_data=em_data, root_input_dir=data_path)

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
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10)

        if args.consensus is True and args.masking is False:
            my_name = f"adamt-mito-source-mitoem-target-{em_data}-consensus-weighting"
        elif args.consensus is True and args.masking is True:
            my_name = f"adamt-mito-source-mitoem-target-{em_data}-consensus-masking"
        else:
            my_name = f"adamt-mito-source-mitoem-target-{em_data}"

        trainer = AdaMTTrainer(
            name=my_name,
            save_root=args.save_root,
            source_train_loader=source_train_loader,
            target_train_loader=target_train_loader,
            val_loader=target_val_loader,
            model=model,
            optimizer=optimizer,
            loss=DummyLoss(),
            metric=DummyLoss(),
            device=device,
            lr_scheduler=scheduler,
            logger=AdaMTLogger,
            mixed_precision=True,
            compile_model=False,
            log_image_interval=1000,
            do_consensus_masking=args.masking
        )

        n_iterations = 100000
        trainer.fit(n_iterations, overwrite_training=False)

        torch.cuda.empty_cache()


def do_adamt_predictions(args, device, data_path: str, pred_path: str):
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
                if args.consensus is True and args.masking is False:
                    model_save_dir = f"adamt-livecell-source-{src}-target-{trg}-consensus-weighting/best.pt"
                elif args.masking is True and args.masking is True:
                    model_save_dir = f"adamt-livecell-source-{src}-target-{trg}-consensus-masking/best.pt"
                else:
                    model_save_dir = f"adamt-livecell-source-{src}-target-{trg}/best.pt"

                if os.path.exists(model_save_dir) is False:
                    print("The model couldn't be found/hasn't been trained yet")
                    continue

                model_state = torch.load(model_save_dir, map_location=torch.device('cpu'))["model_state"]
                model.load_state_dict(model_state)
                model.to(device)

                input_path = data_path + f"images/livecell_test_images/{trg}*"
                output_path = pred_path + f"adamt/source-{src}-target-{trg}/"

                punet_prediction(
                    input_image_path=input_path,
                    output_pred_path=output_path,
                    model=model,
                    device=device,
                    prior_samples=16
                )


def do_adamt_evaluations(data_path: str, pred_path: str):
    cell_list = ['A172', 'BT474', 'BV2', 'Huh7', 'MCF7', 'SHSY5Y', 'SkBr3', 'SKOV3']
    for trg in cell_list:
        gt_path = data_path + f"annotations/livecell_test_images/{trg}/"
        for src in cell_list:
            if src != trg:
                punet_pred_path = pred_path + f"adamt/source-{src}-target-{trg}/"

                if os.path.exists(punet_pred_path) is False:
                    print("The model predictions haven't been generated, hence no evaluation")
                    continue

                run_dice_evaluation(gt_path, punet_pred_path)
                print(f"dice for {trg} from {src}-{trg}")


def main(args):
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        print("Training PUNet on AdaMT framework on MitoEM datasets")
        do_adamt_training(args, data_path=args.data, device=device, em_types=args.type)

    if args.predict:
        print("Getting predictions on MitoEM datasets from the trained AdaMT framework")
        do_adamt_predictions(args, data_path=args.data, pred_path=args.pred_path, device=device)

    if args.evaluate:
        print("Evaluating the AdaMT predictions of MitoEM datasets")
        do_adamt_evaluations(data_path=args.data, pred_path=args.pred_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action='store_true', help="Enables AdaMT-based training on MitoEM datasets")
    parser.add_argument("--predict", action='store_true', help="Obtains AdaMT predictions on MitoEM test-sets")
    parser.add_argument("--evaluate", action='store_true', help="Evaluates AdaMT predictions")

    parser.add_argument("--consensus", action='store_true', help="Activates Consensus (Weighting) in the network")
    parser.add_argument("--masking", action='store_true', help="Uses Consensus Masking in the training")

    parser.add_argument(
        "--data", type=str, default="/mnt/lustre-grete/usr/u16934/data",
        help="Path where the dataset already exists/will be downloaded by the dataloader"
    )
    parser.add_argument(
        "--source_checkpoints", type=str, default="checkpoints",
        help="Path where the pretrained source model checkpoints already exists"
    )
    parser.add_argument(
        "--save_root", default="/mnt/lustre-grete/usr/u16934/models", type=str,
        help="Path where the trained models are stored."
    )
    parser.add_argument(
        "--pred_path", type=str, default="/mnt/lustre-grete/usr/u16934/experiments/pda/source-mitoem",
        help="Path where predictions will be saved."
    )
    parser.add_argument(
        "--type", type=str, default=None,
        help="Whether to run AdaMT on a particular type of data domain."
    )

    args = parser.parse_args()
    main(args)
