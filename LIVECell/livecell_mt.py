import os
import argparse

import torch
from torchvision import transforms
from torch_em.transform.raw import get_raw_transform, AdditiveGaussianNoise, GaussianBlur

from prob_utils.my_models import ProbabilisticUnet
from prob_utils.my_predictions import punet_prediction
from prob_utils.my_evaluations import run_dice_evaluation
from prob_utils.my_datasets import get_dual_livecell_loader
from prob_utils.my_utils import my_standardize_torch, DummyLoss
from prob_utils.my_trainer import MeanTeacherTrainer, MeanTeacherLogger


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


def get_livecell_loaders(path: str, ctype: list, patch_shape=(512, 512), my_augs=my_weak_augmentations()):

    train_loader = get_dual_livecell_loader(
        path=path,
        binary=True,
        split='train',
        patch_shape=patch_shape,
        batch_size=2,
        cell_types=ctype,
        augmentation1=my_augs,
        augmentation2=my_augs,
        download=True
    )

    val_loader = get_dual_livecell_loader(
        path=path,
        binary=True,
        split='val',
        patch_shape=patch_shape,
        batch_size=1,
        cell_types=ctype,
        augmentation1=my_augs,
        augmentation2=my_augs,
        download=True
    )

    return train_loader, val_loader


def do_mean_teacher_training(args, device, data_path: str, source_ckpt_path: str):
    cell_types = ['A172', 'BT474', 'BV2', 'Huh7', 'MCF7', 'SHSY5Y', 'SkBr3', 'SKOV3']

    for trg in cell_types:
        for src in cell_types:
            if src != trg:
                print(f"Transferring {src} network learnings on {trg} using Mean-Teacher")

                my_ckpt = source_ckpt_path + f"punet-source-livecell-{src}/best.pt"
                if os.path.exists(my_ckpt) is False:
                    print("The checkpoint directory couldn't be found/source network hasn't been trained")
                    continue

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
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10)
                model.to(device)

                if args.consensus is True and args.masking is False:
                    my_name = f"mean-teacher-livecell-source-{src}-target-{trg}-consensus-weighting"
                elif args.masking is True and args.masking is True:
                    my_name = f"mean-teacher-livecell-source-{src}-target-{trg}-consensus-masking"
                else:
                    my_name = f"mean-teacher-livecell-source-{src}-target-{trg}"

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
                    log_image_interval=10,
                    do_consensus_masking=args.masking
                )

                n_iterations = 10000
                trainer.fit(n_iterations)


def do_mean_teacher_predictions(args, device, data_path: str, pred_path: str):

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
                    model_save_dir = f"mean-teacher-livecell-source-{src}-target-{trg}-consensus-weighting/best.pt"
                elif args.masking is True and args.masking is True:
                    model_save_dir = f"mean-teacher-livecell-source-{src}-target-{trg}-consensus-masking/best.pt"
                else:
                    model_save_dir = f"mean-teacher-livecell-source-{src}-target-{trg}/best.pt"

                if os.path.exists(model_save_dir) is False:
                    print("The model couldn't be found/hasn't been trained yet")
                    continue

                model_state = torch.load(model_save_dir, map_location=torch.device('cpu'))["teacher_state"]
                model.load_state_dict(model_state)
                model.to(device)

                input_path = data_path + f"images/livecell_test_images/{trg}*"
                output_path = pred_path + f"mean_teacher/source-{src}-target-{trg}/"

                punet_prediction(
                    input_image_path=input_path,
                    output_pred_path=output_path,
                    model=model,
                    device=device,
                    prior_samples=16,
                )


def do_mean_teacher_evaluations(data_path: str, pred_path: str):
    cell_list = ['A172', 'BT474', 'BV2', 'Huh7', 'MCF7', 'SHSY5Y', 'SkBr3', 'SKOV3']

    for trg in cell_list:
        gt_path = data_path + f"annotations/livecell_test_images/{trg}/"

        for src in cell_list:
            if src != trg:
                punet_pred_path = pred_path + f"mean_teacher/source-{src}-target-{trg}/"

                if os.path.exists(punet_pred_path) is False:
                    print("The model predictions haven't been generated, hence no evaluation")
                    continue

                run_dice_evaluation(gt_path, punet_pred_path)
                print(f"dice for {trg} from {src}-{trg}")


def main(args):
    try:
        print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    except AssertionError:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        print("Training PUNet on Mean-Teacher framework on LiveCELL dataset")
        do_mean_teacher_training(args, data_path=args.data, source_ckpt_path=args.source_checkpoints, device=device)

    if args.predict:
        print("Getting predictions on LiveCELL dataset from the trained Mean-Teacher framework")
        do_mean_teacher_predictions(args, data_path=args.data, pred_path=args.pred_path, device=device)

    if args.evaluate:
        print("Evaluating the Mean-Teacher predictions of LiveCELL dataset")
        do_mean_teacher_evaluations(data_path=args.data, pred_path=args.pred_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action='store_true', help="Enables Mean-Teacher training on LiveCELL dataset")
    parser.add_argument("--predict", action='store_true', help="Obtains Mean-Teacher predictions on LiveCELL test-set")
    parser.add_argument("--evaluate", action='store_true', help="Evaluates Mean-Teacher predictions")

    parser.add_argument("--consensus", action='store_true', help="Activates Consensus (Weighting) in the network")
    parser.add_argument("--masking", action='store_true', help="Uses Consensus Masking in the training")

    parser.add_argument(
        "--data", type=str, default="~/data/livecell/",
        help="Path where the dataset already exists/will be downloaded by the dataloader"
    )
    parser.add_argument(
        "--source_checkpoints", type=str, default="checkpoints/",
        help="Path where the dataset already exists/will be downloaded by the dataloader"
    )
    parser.add_argument(
        "--pred_path", type=str, default="~/predictions/livecell/", help="Path where predictions will be saved"
    )

    args = parser.parse_args()
    main(args)
