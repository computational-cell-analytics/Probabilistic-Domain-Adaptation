import os
import argparse

import torch
from torchvision import transforms

from torch_em.data import MinForegroundSampler
from torch_em.transform.raw import get_raw_transform, AdditiveGaussianNoise, GaussianBlur

from prob_utils.my_models import ProbabilisticUnet
from prob_utils.my_predictions import punet_prediction
from prob_utils.my_evaluations import run_em_dice_evaluation
from prob_utils.my_utils import my_standardize_torch, DummyLoss
from prob_utils.my_trainer import MeanTeacherTrainer, MeanTeacherLogger
from prob_utils.my_datasets import get_vnc_mito_loader, get_lucchi_loader


def my_weak_augmentations(p=0.25):
    norm = my_standardize_torch
    aug1 = transforms.Compose([
        my_standardize_torch,
        transforms.RandomApply([GaussianBlur()], p=p),
        transforms.RandomApply([AdditiveGaussianNoise(scale=(0, 0.15), clip_kwargs=False)], p=p),
    ])
    return get_raw_transform(
        normalizer=norm,
        augmentation1=aug1
    )


def get_dual_loaders(
    em_data: str,
    root_input_dir: str,
    patch_shape=(1, 256, 256),
    my_augs=my_weak_augmentations(),
    my_sampler=MinForegroundSampler(min_fraction=0.05)
):
    path = root_input_dir + f"{em_data}"

    if em_data == "vnc":
        train_loader = get_vnc_mito_loader(
            path=path,
            partition="tr",
            batch_size=2,
            patch_shape=patch_shape,
            ndim=2,
            binary=True,
            sampler=my_sampler,
            augmentation1=my_augs,
            augmentation2=my_augs,
            download=True
        )
        val_loader = get_vnc_mito_loader(
            path=path,
            partition="ts",
            batch_size=1,
            patch_shape=patch_shape,
            ndim=2,
            binary=True,
            sampler=my_sampler,
            augmentation1=my_augs,
            augmentation2=my_augs,
            download=True
        )

    elif em_data == "lucchi":
        train_loader = get_lucchi_loader(
            path=path,
            split="train",
            batch_size=4,
            ndim=2,
            patch_shape=patch_shape,
            sampler=my_sampler,
            augmentation1=my_augs,
            augmentation2=my_augs,
            download=True
        )
        val_loader = get_lucchi_loader(
            path=path,
            split="test",
            batch_size=1,
            ndim=2,
            patch_shape=patch_shape,
            sampler=my_sampler,
            augmentation1=my_augs,
            augmentation2=my_augs,
            download=True
        )

    return train_loader, val_loader


def do_mean_teacher_training(args, device, source_ckpt_path: str):
    em_types = ["vnc", "lucchi"]

    for em_data in em_types:
        print(f"Training on {em_data} using Mean-Teacher scheme")

        train_loader, val_loader = get_dual_loaders(em_data=em_data)

        my_ckpt = source_ckpt_path + "punet-source-mitoemv2/best.pt"
        if os.path.exists(my_ckpt) is False:
            print("The checkpoint directory couldn't be found/source network hasn't been trained")
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
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.9, patience=10, verbose=True
        )
        model.to(device)

        if args.consensus is True and args.masking is False:
            my_name = f"mean-teacher-lung-source-mitoemv2-target-{em_data}-consensus-weighting"
        elif args.masking is True and args.masking is True:
            my_name = f"mean-teacher-lung-source-mitoemv2-target-{em_data}-consensus-masking"
        else:
            my_name = f"mean-teacher-lung-source-mitoemv2-target-{em_data}"

        trainer = MeanTeacherTrainer(
            name=my_name,
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
            log_image_interval=1000,
            do_consensus_masking=args.masking
        )

        n_iterations = 10000
        trainer.fit(n_iterations)


def do_mean_teacher_predictions(device, data_path: str, pred_path: str):
    em_types = ["vnc", "lucchi"]

    model = ProbabilisticUnet(
        input_channels=1,
        num_classes=1,
        num_filters=[64, 128, 256, 512],
        latent_dim=6,
        no_convs_fcomb=3,
        beta=1.0,
        rl_swap=True
    )

    for em_data in em_types:
        if args.consensus is True and args.masking is False:
            model_save_dir = f"mean-teacher-lung-source-mitoemv2-target-{em_data}-consensus-weighting/best.pt"
        elif args.masking is True and args.masking is True:
            model_save_dir = f"mean-teacher-lung-source-mitoemv2-target-{em_data}-consensus-masking/best.pt"
        else:
            model_save_dir = f"mean-teacher-lung-source-mitoemv2-target-{em_data}/best.pt"

        if os.path.exists(model_save_dir) is False:
            print("The model couldn't be found/hasn't been trained yet")
            continue

        model_state = torch.load(model_save_dir, map_location=torch.device('cpu'))["model_state"]
        model.load_state_dict(model_state)
        model.to(device)

        output_path = pred_path + f"mean_teacher/source-mitoemv2-target-{em_data}/"

        if em_data == "lucchi":
            input_path = data_path + "lucchi/Lucchi++/Test_In/*"
        elif em_data == "vnc":
            input_path = data_path + "vnc/groundtruth-drosophila-vnc-master/stack1/raw/*"

        punet_prediction(input_image_path=input_path, output_pred_path=output_path, model=model, device=device)


def do_mean_teacher_evaluations(data_path: str, pred_path: str):
    em_types = ["vnc", "lucchi"]

    for em_data in em_types:
        root_output = pred_path + "punet_predictions/"

        if em_data == "lucchi":
            gt_path = data_path + "lucchi/Lucchi++/Test_Out/"
            output_path = root_output + "lucchi"
        elif em_data == "vnc":
            gt_path = data_path + "vnc/groundtruth-drosophila-vnc-master/stack1/mitochondria/"
            output_path = root_output + "vnc/"

        run_em_dice_evaluation(gt_f_path=gt_path, pred_path=output_path, model=em_data)


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
        do_mean_teacher_evaluations(data_path=args.data, pred_path=args.pred_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action='store_true', help="Enables Mean-Teacher training on MitoEM datasets")
    parser.add_argument("--predict", action='store_true', help="Obtains Mean-Teacher predictions on MitoEM test-set")
    parser.add_argument("--evaluate", action='store_true', help="Evaluates Mean-Teacher predictions")

    parser.add_argument("--consensus", action='store_true', help="Activates Consensus (Weighting) in the network")
    parser.add_argument("--masking", action='store_true', help="Uses Consensus Masking in the training")

    parser.add_argument("--data", default="~/data/", type=str, help="Path where the dataset already exists for MitoEM")
    parser.add_argument(
        "--source_checkpoints", type=str, default="checkpoints/",
        help="Path where the dataset already exists/will be downloaded by the dataloader"
    )
    parser.add_argument(
        "--pred_path", type=str, default="~/predictions/mitoem/", help="Path where predictions will be saved"
    )

    args = parser.parse_args()
    main(args)
