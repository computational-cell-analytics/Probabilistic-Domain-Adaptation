import os
import argparse

import torch
from torchvision import transforms

from torch_em.transform.raw import get_raw_transform, AdditiveGaussianNoise, GaussianBlur

from prob_utils.my_models import ProbabilisticUnet
from prob_utils.my_predictions import punet_prediction
from prob_utils.my_trainer import AdaMTTrainer, AdaMTLogger
from prob_utils.my_evaluations import run_lung_dice_evaluation
from prob_utils.my_utils import DummyLoss, my_standardize_torch
from prob_utils.my_datasets import get_jsrt_s1_loader, get_jsrt_s2_loader, get_montgomery_loader, get_nih_loader


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


def get_source_lung_loaders(lung_domain_name: str, root_input_dir: str):
    print(f"Getting the Source Loader of Domain - {lung_domain_name}..")

    if lung_domain_name == "jsrt1":
        source_train_loader = get_jsrt_s1_loader(
            data_path=root_input_dir + "jsrt1",
            split="train",
            batch_size=2,
            val_fraction=0
        )

    elif lung_domain_name == "jsrt2":
        source_train_loader = get_jsrt_s2_loader(
            data_path=root_input_dir + "jsrt2",
            split="train",
            batch_size=2,
            val_fraction=0
        )

    elif lung_domain_name == "nih":
        source_train_loader = get_nih_loader(
            data_path=root_input_dir + "nih_processed/",
            split="train",
            batch_size=2,
            val_fraction=0
        )

    elif lung_domain_name == "montgomery":
        source_train_loader = get_montgomery_loader(
            data_path=root_input_dir + "montgomery_processed/",
            split="train",
            batch_size=2,
            val_fraction=0
        )

    return source_train_loader


def get_target_lung_loaders(lung_domain_name: str, root_input_dir: str, my_augs=my_weak_augmentations()):
    print(f"Getting the Target Loader of Domain - {lung_domain_name}..")

    if lung_domain_name == "jsrt1":
        train_loader = get_jsrt_s1_loader(
            data_path=root_input_dir + "jsrt1",
            split="train",
            batch_size=2,
            augmentation1=my_augs,
            augmentation2=my_augs
        )
        val_loader = get_jsrt_s1_loader(
            data_path=root_input_dir + "jsrt1",
            split="val",
            batch_size=1,
            augmentation1=my_augs,
            augmentation2=my_augs
        )

    elif lung_domain_name == "jsrt2":
        train_loader = get_jsrt_s2_loader(
            data_path=root_input_dir + "jsrt2",
            split="train",
            batch_size=2,
            augmentation1=my_augs,
            augmentation2=my_augs
        )
        val_loader = get_jsrt_s2_loader(
            data_path=root_input_dir + "jsrt2",
            split="val",
            batch_size=1,
            augmentation1=my_augs,
            augmentation2=my_augs
        )

    elif lung_domain_name == "nih":
        train_loader = get_nih_loader(
            data_path=root_input_dir + "nih_processed/",
            split="train",
            batch_size=2,
            augmentation1=my_augs,
            augmentation2=my_augs
        )
        val_loader = get_nih_loader(
            data_path=root_input_dir + "nih_processed/",
            split="val",
            batch_size=1,
            augmentation1=my_augs,
            augmentation2=my_augs
        )

    elif lung_domain_name == "montgomery":
        train_loader = get_montgomery_loader(
            data_path=root_input_dir + "montgomery_processed/",
            split="train",
            batch_size=2,
            augmentation1=my_augs,
            augmentation2=my_augs
        )
        val_loader = get_montgomery_loader(
            data_path=root_input_dir + "montgomery_processed/",
            split="val",
            batch_size=1,
            augmentation1=my_augs,
            augmentation2=my_augs
        )

    return train_loader, val_loader


def do_adamt_training(args, device, data_path: str):
    lung_list = ["jsrt1", "jsrt2", "nih", "montgomery"]

    for source_domain in lung_list:
        for target_domain in lung_list:
            if source_domain != target_domain:
                print(f"Training Lung X-Rays from Source - {source_domain} to Target - {target_domain}")

                source_train_loader = get_source_lung_loaders(lung_domain_name=source_domain, root_input_dir=data_path)
                target_train_loader, target_val_loader = get_target_lung_loaders(
                    lung_domain_name=target_domain, root_input_dir=data_path
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

                if args.consensus is True and args.masking is False:
                    my_name = f"adamt-lung-source-{source_domain}-target-{target_domain}-consensus-weighting"
                elif args.masking is True and args.masking is True:
                    my_name = f"adamt-lung-source-{source_domain}-target-{target_domain}-consensus-masking"
                else:
                    my_name = f"adamt-lung-source-{source_domain}-target-{target_domain}"

                trainer = AdaMTTrainer(
                    name=my_name,
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
                    log_image_interval=1000,
                    do_consensus_masking=args.masking
                )

                n_iterations = 100000
                trainer.fit(n_iterations)


def do_adamt_predictions(device, data_path: str, pred_path: str):
    lung_domain_names = ["jsrt1", "jsrt2", "nih", "montgomery"]

    model = ProbabilisticUnet(
        input_channels=1,
        num_classes=1,
        num_filters=[64, 128, 256, 512],
        latent_dim=6,
        no_convs_fcomb=3,
        beta=1.0,
        rl_swap=True
    )

    for source_lung_domain in lung_domain_names:
        for target_lung_domain in lung_domain_names:
            if source_lung_domain != target_lung_domain:

                if args.consensus is True and args.masking is False:
                    model_save_dir = f"adamt-lung-source-{source_lung_domain}-target-{target_lung_domain}-" + \
                        "consensus-weighting/best.pt"
                elif args.masking is True and args.masking is True:
                    model_save_dir = f"adamt-lung-source-{source_lung_domain}-target-{target_lung_domain}-" + \
                        "consensus-masking/best.pt"
                else:
                    model_save_dir = f"adamt-lung-source-{source_lung_domain}-target-{target_lung_domain}/best.pt"

                if os.path.exists(model_save_dir) is False:
                    print("The model couldn't be found/hasn't been trained yet")
                    continue

                model_state = torch.load(model_save_dir, map_location=torch.device('cpu'))["model_state"]
                model.load_state_dict(model_state)
                model.to(device)

                output_path = pred_path + f"adamt/source-{source_lung_domain}-target-{target_lung_domain}/"

                if target_lung_domain == "jsrt1":
                    input_path = data_path + "jsrt1/test/org/*"
                elif target_lung_domain == "jsrt2":
                    input_path = data_path + "jsrt2/segmentation/org_test/*"
                elif target_lung_domain == "nih":
                    input_path = data_path + "nih_processed/images/test/*"
                elif target_lung_domain == "montgomery":
                    input_path = data_path + "montgomery_processed/images/test/*"

                punet_prediction(input_image_path=input_path, output_pred_path=output_path, model=model, device=device)


def do_adamt_evaluations(data_path: str, pred_path: str):
    lung_domain_names = ["jsrt1", "jsrt2", "nih", "montgomery"]

    for source_lung_domain in lung_domain_names:
        for target_lung_domain in lung_domain_names:
            if source_lung_domain != target_lung_domain:
                punet_pred_path = pred_path + f"adamt/source-{source_lung_domain}-target-{target_lung_domain}/"

                if os.path.exists(punet_pred_path) is False:
                    print("The source model predictions couldn't be found/haven't been generated")
                    continue

                if target_lung_domain == "jsrt1":
                    gt_path = data_path + "jsrt/test/label/"
                elif target_lung_domain == "jsrt2":
                    gt_path = data_path + "jsrt2/segmentation/label_test/"
                elif target_lung_domain == "nih":
                    gt_path = data_path + "nih_processed/labels/test/"
                elif target_lung_domain == "montgomery":
                    gt_path = data_path + "montgomery_processed/labels/test/"

                run_lung_dice_evaluation(gt_path, punet_pred_path, lung_domain=target_lung_domain)

                print(f"Dice on {target_lung_domain} from {source_lung_domain}")


def main(args):
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        print("Training PUNet on AdaMT framework on Lung X-Ray datasets")
        do_adamt_training(args, data_path=args.data, device=device)

    if args.predict:
        print("Getting predictions on Lung X-Ray datasets from the trained AdaMT framework")
        do_adamt_predictions(args, data_path=args.data, pred_path=args.pred_path, device=device)

    if args.evaluate:
        print("Evaluating the AdaMT predictions of Lung X-Ray dataset")
        do_adamt_evaluations(data_path=args.data, pred_path=args.pred_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action='store_true', help="Enables AdaMT-based training on Lung X-Ray datasets")
    parser.add_argument("--predict", action='store_true', help="Obtains AdaMT predictions on Lung X-Ray test-set")
    parser.add_argument("--evaluate", action='store_true', help="Evaluates AdaMT predictions")

    parser.add_argument("--consensus", action='store_true', help="Activates Consensus (Weighting) in the network")
    parser.add_argument("--masking", action='store_true', help="Uses Consensus Masking in the training")

    parser.add_argument("--data", type=str, help="Path where the dataset already exists for Lung X-Rays")
    parser.add_argument(
        "--pred_path", type=str, default="~/predictions/livecell/", help="Path where predictions will be saved"
    )

    args = parser.parse_args()
    main(args)
