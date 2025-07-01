import os
import argparse
from glob import glob

import numpy as np
import imageio.v3 as imageio

import torch
from torchvision import transforms

from torch_em.transform.raw import get_raw_transform, AdditiveGaussianNoise, GaussianBlur, RandomContrast

from prob_utils.my_models import ProbabilisticUnet
from prob_utils.my_predictions import punet_prediction
from prob_utils.my_evaluations import run_dice_evaluation
from prob_utils.my_datasets import get_dual_livecell_loader
from prob_utils.my_utils import my_standardize_torch, DummyLoss
from prob_utils.my_trainer import FixMatchTrainer, FixMatchLogger


def compute_class_distribution(root_folder):
    bg_list, fg_list = [], []
    total = 0

    files = glob(os.path.join(root_folder, "*"))
    assert len(files) > 0, f"Did not find predictions @ {root_folder}"

    for pl_path in files:
        img = imageio.imread(pl_path)
        img = np.where(img >= 0.5, 1, 0)
        _, counts = np.unique(img, return_counts=True)
        assert len(counts) == 2
        bg_list.append(counts[0])
        fg_list.append(counts[1])
        total += img.size

    bg_frequency = sum(bg_list) / float(total)
    fg_frequency = sum(fg_list) / float(total)
    assert np.isclose(bg_frequency + fg_frequency, 1.0)
    return [bg_frequency, fg_frequency]


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


def my_strong_augmentations(p=0.9):
    norm = my_standardize_torch
    aug1 = transforms.Compose([
        my_standardize_torch,
        transforms.RandomApply([GaussianBlur(sigma=(1.0, 4.0))], p=p),
        transforms.RandomApply([AdditiveGaussianNoise(scale=(0.1, 0.35), clip_kwargs=False)], p=p),
        transforms.RandomApply([RandomContrast(alpha=(0.33, 3), mean=0.0, clip_kwargs=False)], p=p),
    ])
    return get_raw_transform(
        normalizer=norm,
        augmentation1=aug1
    )


def get_livecell_loaders(
    path: str,
    ctype: list,
    patch_shape=(512, 512),
    my_weak_augs=my_weak_augmentations(),
    my_strong_augs=my_strong_augmentations()
):
    train_loader = get_dual_livecell_loader(
        path=path,
        binary=True,
        split='train',
        patch_shape=patch_shape,
        batch_size=2,
        cell_types=ctype,
        augmentation1=my_weak_augs,
        augmentation2=my_strong_augs,
        download=True
    )

    val_loader = get_dual_livecell_loader(
        path=path,
        binary=True,
        split='val',
        patch_shape=patch_shape,
        batch_size=1,
        cell_types=ctype,
        augmentation1=my_weak_augs,
        augmentation2=my_strong_augs,
        download=True
    )

    return train_loader, val_loader


def do_fixmatch_training(
    args, device, data_path: str, source_ckpt_path: str, pseudo_labels: str, use_distro_alignment=True
):
    cell_types = ['A172', 'BT474', 'BV2', 'Huh7', 'MCF7', 'SHSY5Y', 'SkBr3', 'SKOV3']

    for trg in cell_types:
        for src in cell_types:
            if src != trg:
                print(f"Transferring {src} network learnings on {trg} using FixMatch")

                my_ckpt = source_ckpt_path + f"punet-source-livecell-{src}/best.pt"
                if os.path.exists(my_ckpt) is False:
                    print("The checkpoint directory couldn't be found/source network hasn't been trained")
                    continue

                if use_distro_alignment:
                    print(f"Getting scores for source-{src} at targets-{trg}")
                    pred_folder = pseudo_labels + f"punet_source_predictions/{src}/annotations/livecell_train_val_images/{trg}"
                    src_dist = compute_class_distribution(pred_folder)
                else:
                    src_dist = None

                train_loader, val_loader = get_livecell_loaders(path=data_path, ctype=[trg])

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
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10)
                model.to(device)

                if args.consensus is True and args.masking is False:
                    my_name = f"fixmatch-livecell-source-{src}-target-{trg}-consensus-weighting"
                elif args.consensus is True and args.masking is True:
                    my_name = f"fixmatch-livecell-source-{src}-target-{trg}-consensus-masking"
                else:
                    my_name = f"fixmatch-livecell-source-{src}-target-{trg}"

                trainer = FixMatchTrainer(
                    name=my_name,
                    ckpt_model=my_ckpt,
                    source_distribution=src_dist,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    model=model,
                    optimizer=optimizer,
                    loss=DummyLoss(),
                    metric=DummyLoss(),
                    device=device,
                    lr_scheduler=scheduler,
                    logger=FixMatchLogger,
                    mixed_precision=True,
                    log_image_interval=10,
                    do_consensus_masking=args.masking
                )

                n_iterations = 10000
                trainer.fit(n_iterations)


def do_fixmatch_predictions(args, device, data_path: str, pred_path: str):
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
                    model_save_dir = f"fixmatch-livecell-source-{src}-target-{trg}-consensus-weighting/best.pt"
                elif args.masking is True and args.masking is True:
                    model_save_dir = f"fixmatch-livecell-source-{src}-target-{trg}-consensus-masking/best.pt"
                else:
                    model_save_dir = f"fixmatch-livecell-source-{src}-target-{trg}/best.pt"
                
                if os.path.exists(model_save_dir) is False:
                    print("The model couldn't be found/hasn't been trained yet")
                    continue

                model_state = torch.load(
                    model_save_dir, map_location=torch.device('cpu'), weights_only=False
                )["model_state"]
                model.load_state_dict(model_state)
                model.to(device)

                input_path = data_path + f"images/livecell_test_images/{trg}*"
                output_path = pred_path + f"fixmatch/source-{src}-target-{trg}/"

                punet_prediction(input_image_path=input_path, output_pred_path=output_path, model=model, device=device, prior_samples=16)


def do_fixmatch_evaluations(data_path: str, pred_path: str):
    cell_list = ['A172', 'BT474', 'BV2', 'Huh7', 'MCF7', 'SHSY5Y', 'SkBr3', 'SKOV3']
    for trg in cell_list:
        gt_path = data_path + f"annotations/livecell_test_images/{trg}/"
        for src in cell_list:
            if src != trg:
                punet_pred_path = pred_path + f"fixmatch/source-{src}-target-{trg}/"

                if os.path.exists(punet_pred_path) is False:
                    print("The model predictions haven't been generated, hence no evaluation")
                    continue

                run_dice_evaluation(gt_path, punet_pred_path)
                print(f"dice for {trg} from {src}-{trg}")


def main(args):
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        print("Training PUNet on Fixmatch framework on LiveCELL dataset")
        do_fixmatch_training(args, data_path=args.data, source_ckpt_path=args.source_checkpoints, pseudo_labels=args.pred_path, device=device)

    if args.predict:
        print("Getting predictions on LiveCELL dataset from the trained Fixmatch framework")
        do_fixmatch_predictions(args, data_path=args.data, pred_path=args.pred_path, device=device)

    if args.evaluate:
        print("Evaluating the Fixmatch predictions of LiveCELL dataset")
        do_fixmatch_evaluations(data_path=args.data, pred_path=args.pred_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action='store_true', help="Enables Fixmatch training on LiveCELL dataset")
    parser.add_argument("--predict", action='store_true', help="Obtains Fixmatch predictions on LiveCELL test-set")
    parser.add_argument("--evaluate", action='store_true', help="Evaluates Fixmatch predictions")

    parser.add_argument("--consensus", action='store_true', help="Activates Consensus (Weighting) in the network")
    parser.add_argument("--masking", action='store_true', help="Uses Consensus Masking in the training")

    parser.add_argument("--data", type=str, default="~/data/livecell/", help="Path where the dataset already exists/will be downloaded by the dataloader")
    parser.add_argument("--source_checkpoints", type=str, default="checkpoints/", help="Path where the dataset already exists/will be downloaded by the dataloader")
    parser.add_argument("--pred_path", type=str, default="~/predictions/livecell/", help="Path where predictions will be saved")

    args = parser.parse_args()
    main(args)
