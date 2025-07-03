import numpy as np
from skimage import exposure
import imageio.v3 as imageio
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

import torch_em
from torch_em.data.datasets.light_microscopy.livecell import get_livecell_paths
from torch_em.transform.raw import get_raw_transform, AdditiveGaussianNoise, GaussianBlur, RandomContrast

from prob_utils.my_models import ProbabilisticUnet
from prob_utils.my_utils import my_standardize_torch


def _enhance_image(im, do_norm=False):
    # apply CLAHE to improve the image quality
    if do_norm:
        im -= im.min(axis=(0, 1), keepdims=True)
        im /= (im.max(axis=(0, 1), keepdims=True) + 1e-6)

    im = exposure.equalize_adapthist(im)
    im *= 255
    return im


# Some augmentation stuff
def my_weak_augmentations(p=0.25):
    norm = my_standardize_torch
    aug1 = transforms.Compose([
        my_standardize_torch,
        transforms.RandomApply([GaussianBlur()], p=p),
        transforms.RandomApply([AdditiveGaussianNoise(scale=(0, 0.15), clip_kwargs=False)], p=p),
    ])
    return get_raw_transform(normalizer=norm, augmentation1=aug1)


def my_strong_augmentations(p=0.9):
    norm = my_standardize_torch
    aug1 = transforms.Compose([
        my_standardize_torch,
        transforms.RandomApply([GaussianBlur(sigma=(0.6, 3.0))], p=p),
        transforms.RandomApply([AdditiveGaussianNoise(scale=(0.05, 0.25), clip_kwargs=False)], p=p/2),
        transforms.RandomApply([RandomContrast(mean=0.0, alpha=(0.33, 3.0), clip_kwargs=False)], p=p),
    ])
    return get_raw_transform(normalizer=norm, augmentation1=aug1)


def main():
    # Stuff for inference
    cell_type = "BV2"

    # Get the image and corresponding labels.
    image_paths, gt_paths = get_livecell_paths(
        path="/mnt/lustre-grete/usr/u16934/data/livecell", split="test", cell_types=[cell_type]
    )

    # Prepare the model
    model = ProbabilisticUnet(
        input_channels=1,
        num_classes=1,
        num_filters=[64, 128, 256, 512],
        latent_dim=6,
        no_convs_fcomb=3,
        beta=1.0,
        rl_swap=True
    )

    # Load the model with pretrained weights
    model.load_state_dict(
        torch.load(
            f"../LIVECell/checkpoints/punet-source-livecell-{cell_type}/best.pt",
            weights_only=False,
            map_location="cpu",
        )["model_state"]
    )

    model.to("cuda")
    model.eval()

    # More stuff for inference
    prior_samples = 8
    activation = torch.nn.Sigmoid()

    for i, (ipath, gpath) in enumerate(zip(image_paths, gt_paths), start=1):
        # I choose a specific image because it's pretty! :)
        if cell_type == "BV2" and i != 3:
            continue

        # Load image and labels
        og_image = imageio.imread(ipath)
        gt = imageio.imread(gpath)
        gt = (gt > 0).astype("uint8")  # And binarize the labels.

        # Pass the inputs through the model.
        image = torch_em.transform.raw.standardize(og_image)
        image = torch.from_numpy(image)[None, None].to("cuda")

        model.forward(image, None, training=False)  # Forward pass for PUNet

        # Aggregate samples from the prior
        samples_per_patch = [activation(model.sample(testing=True)) for _ in range(prior_samples)]
        pred = torch.stack(samples_per_patch, dim=0).sum(dim=0) / prior_samples
        pred = pred.detach().cpu().numpy().squeeze()

        # Calculate the consensus
        upper_thres, lower_thres = 0.9, 0.1
        consensus = [
            torch.where(
                (my_sample >= upper_thres) + (my_sample <= lower_thres),
                torch.tensor(1.).to("cuda"),
                torch.tensor(0.).to("cuda")
            ) for my_sample in samples_per_patch
        ]
        consensus = torch.stack(consensus, dim=0).sum(dim=0) / prior_samples
        consensus = consensus.detach().cpu().numpy().squeeze()

        # Let's plot stuff before storing them for figures
        fig, ax = plt.subplots(1, 5, figsize=(30, 20))
        ax[0].imshow(og_image)
        ax[1].imshow(gt)
        ax[2].imshow(pred)  # pseudo labels
        ax[3].imshow(consensus)  # consensus response

        # Choose one sample and prepare it for prediction figure.
        uno = samples_per_patch[0].squeeze().detach().cpu().numpy()
        uno = np.where(uno > (1 - 1e-7), 0.5, uno)
        ax[4].imshow(uno)  # one sample.
        plt.savefig("./test.png")
        plt.close()

        # Now, let's store each figure one by one.
        def _prepare_figure(im, fpath):
            plt.imshow(im, cmap="gray")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(fpath, bbox_inches="tight", pad_inches=0, dpi=300)
            plt.close()

        _prepare_figure(_enhance_image(og_image), "image.png")
        _prepare_figure(gt, "gt.png")
        _prepare_figure(pred, "pseudo_labels.png")
        _prepare_figure(consensus, "consensus_response.png")

        # Run transformations on image for augmented views
        weak_transform = my_weak_augmentations()
        strong_transform = my_strong_augmentations()

        weak_image = weak_transform(torch.from_numpy(og_image)[None].to(torch.float32)).numpy().squeeze()
        strong_image = strong_transform(torch.from_numpy(og_image)[None].to(torch.float32)).numpy().squeeze()

        _prepare_figure(_enhance_image(weak_image, do_norm=True), "weak_image.png")
        _prepare_figure(_enhance_image(strong_image, do_norm=True), "strong_image.png")

        breakpoint()


if __name__ == "__main__":
    main()
