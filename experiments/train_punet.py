import torch

import torch_em

from common import get_dataloaders, get_model


def train_punet(data_path, dataset_name, patch_shape, **kwargs):
    """Train PUNet model.
    """

    device = "cuda" if torch.cuda.is_available else "cpu"
    train_loader, val_loader = get_dataloaders(
        data_path=data_path, dataset_name=dataset_name, patch_shape=patch_shape, **kwargs
    )
    model = get_model(dataset_name=dataset_name, device=device, backbone="punet")

    name = f"punet-source-{dataset_name}",
    if "cell_types" in kwargs:
        cell_type = kwargs["cell_types"][0]
        name += f"-{cell_type}"

    if "subtype" in kwargs:
        subtype = kwargs["subtype"]
        name += f"-{subtype}"

    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-4,
        log_image_interval=100,
        mixed_precision=True,
        compile_model=False,
    )
    trainer.fit(int(1e5), overwrite_training=False)


def train_em(args):
    # Train source models per mitochondria EM data choices.
    for subtype in ["mitoem", "vnc", "lucchi", "urocell"]:
        train_punet(
            data_path=args.input_path,
            dataset_name="em",
            patch_shape=(16, 512, 512),
            subtype=subtype,
        )


def main(args):
    if args.dataset == "em":
        train_em(args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, default="/mnt/vast-nhr/projects/cidas/cca/data")
    parser.add_argument("-d", "--dataset_name", type=str, required=True)
    parser.add_argument("-s", "--save_root", type=str, default=None)
    args = parser.parse_args()
    main(args)
