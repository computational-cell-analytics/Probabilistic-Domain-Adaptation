import os
import argparse

import torch

from prob_utils.my_utils import DummyLoss
from prob_utils.my_models import ProbabilisticUnet
from prob_utils.my_predictions import punet_prediction
from prob_utils.my_evaluations import run_dice_evaluation
from prob_utils.my_trainer import MeanTeacherTrainer, MeanTeacherLogger

from common import get_dual_loaders, my_weak_augmentations


def do_mean_teacher_training(args, device, data_path: str, source_ckpt_path: str):
    em_types = ["vnc", "lucchi", "urocell"]
    for em_data in em_types:
        print(f"Training on {em_data} using Mean-Teacher scheme")
        train_loader, val_loader = get_dual_loaders(
            em_data=em_data,
            root_input_dir=data_path,
            weak_augs=my_weak_augmentations(),
            strong_augs=my_weak_augmentations(),
        )

        my_ckpt = os.path.join(source_ckpt_path, "punet-source-mitoem", "best.pt")
        if not os.path.exists(my_ckpt):
            print("The checkpoint directory couldn't be found / source network hasn't been trained")
            continue

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

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5)

        if args.consensus is True and args.masking is False:
            my_name = f"mean-teacher-mito-source-mitoem-target-{em_data}-consensus-weighting"
        elif args.consensus is True and args.masking is True:
            my_name = f"mean-teacher-mito-source-mitoem-target-{em_data}-consensus-masking"
        else:
            my_name = f"mean-teacher-mito-source-mitoem-target-{em_data}"

        trainer = MeanTeacherTrainer(
            name=my_name,
            save_root=args.save_root,
            ckpt_teacher=my_ckpt,
            ckpt_model=my_ckpt,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer,
            loss=DummyLoss(),
            metric=DummyLoss(),
            device=device,
            lr_scheduler=scheduler,
            logger=MeanTeacherLogger,
            mixed_precision=True,
            compile_model=False,
            log_image_interval=100,
            do_consensus_masking=args.masking,
        )

        n_iterations = 10000
        trainer.fit(n_iterations, overwrite_training=False)


def do_mean_teacher_predictions(args, device, data_path: str, pred_path: str):
    em_types = ["vnc", "lucchi", "urocell"]
    for em_type in em_types:

        if args.consensus is True and args.masking is False:
            name = f"mean-teacher-mito-source-mitoem-target-{em_type}-consensus-weighting"
            output_path = os.path.join(pred_path, "mean_teacher", f"source-mitoem-target-{em_type}-consensus-weighting")
        elif args.consensus is True and args.masking is True:
            name = f"mean-teacher-mito-source-mitoem-target-{em_type}-consensus-masking"
            output_path = os.path.join(pred_path, "mean_teacher", f"source-mitoem-target-{em_type}-consensus-masking")
        else:
            name = f"mean-teacher-mito-source-mitoem-target-{em_type}"
            output_path = os.path.join(pred_path, "mean_teacher", f"source-mitoem-target-{em_type}")

        model_save_dir = os.path.join(
            ("./" if args.save_root is None else args.save_root), "checkpoints", name, "best.pt"
        )

        if os.path.exists(model_save_dir) is False:
            print("The model couldn't be found / hasn't been trained yet")
            continue

        model = ProbabilisticUnet(
            input_channels=1,
            num_classes=1,
            num_filters=[64, 128, 256, 512],
            latent_dim=6,
            no_convs_fcomb=3,
            beta=1.0,
            rl_swap=True
        )

        model_state = torch.load(model_save_dir, map_location=torch.device('cpu'), weights_only=False)["model_state"]
        model.load_state_dict(model_state)
        model.to(device)

        if em_type == "lucchi":
            input_path = os.path.join(data_path, "lucchi", "Lucchi++", "Test_In", "*")
        elif em_type == "vnc":
            input_path = os.path.join(data_path, "vnc", "groundtruth-drosophila-vnc-master", "stack1", "raw", "*")
        elif em_type == "urocell":
            input_path = os.path.join(data_path, "urocell", "preprocessed", "*_image.tif")

        punet_prediction(input_image_path=input_path, output_pred_path=output_path, model=model, device=device)


def do_mean_teacher_evaluations(args, data_path: str, pred_path: str):
    em_types = ["vnc", "lucchi", "urocell"]
    for em_type in em_types:

        if args.consensus is True and args.masking is False:
            output_path = os.path.join(pred_path, "mean_teacher", f"source-mitoem-target-{em_type}-consensus-weighting")
        elif args.consensus is True and args.masking is True:
            output_path = os.path.join(pred_path, "mean_teacher", f"source-mitoem-target-{em_type}-consensus-masking")
        else:
            output_path = os.path.join(pred_path, "mean_teacher", f"source-mitoem-target-{em_type}")

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
        print("Training PUNet on Mean-Teacher framework on MitoEM datasets")
        do_mean_teacher_training(args, data_path=args.data, source_ckpt_path=args.source_checkpoints, device=device)

    if args.predict:
        print("Getting predictions on MitoEM datasets from the trained Mean-Teacher framework")
        do_mean_teacher_predictions(args, data_path=args.data, pred_path=args.pred_path, device=device)

    if args.evaluate:
        print("Evaluating the Mean-Teacher predictions of MitoEM dataset")
        do_mean_teacher_evaluations(args, data_path=args.data, pred_path=args.pred_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action='store_true', help="Enables Mean-Teacher training on MitoEM datasets")
    parser.add_argument("--predict", action='store_true', help="Obtains Mean-Teacher predictions on MitoEM test-set")
    parser.add_argument("--evaluate", action='store_true', help="Evaluates Mean-Teacher predictions")

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

    args = parser.parse_args()
    main(args)
