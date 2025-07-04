import argparse
import os
from glob import glob
from shutil import copytree

import imageio.v3 as imageio

import numpy as np
from skimage.transform import resize
from tqdm import tqdm

TARGET_SHAPE = (256, 256)


def prepare_montgomery(input_folder, output_folder):
    image_files = glob(os.path.join(input_folder, "CXR_png", "*.png"))
    image_files.sort()
    label_files_left = glob(os.path.join(input_folder, "ManualMask", "leftMask", "*.png"))
    label_files_left.sort()
    label_files_right = glob(os.path.join(input_folder, "ManualMask", "rightMask", "*.png"))
    label_files_right.sort()

    assert len(image_files) > 0
    assert len(image_files) == len(label_files_left) == len(label_files_right)

    out_image_train = os.path.join(output_folder, "images", "train")
    out_image_test = os.path.join(output_folder, "images", "test")
    os.makedirs(out_image_train, exist_ok=True)
    os.makedirs(out_image_test, exist_ok=True)

    out_label_train = os.path.join(output_folder, "labels", "train")
    out_label_test = os.path.join(output_folder, "labels", "test")
    os.makedirs(out_label_train, exist_ok=True)
    os.makedirs(out_label_test, exist_ok=True)

    for im_file, left_file, right_file in tqdm(zip(
        image_files, label_files_left, label_files_right), total=len(image_files)
    ):
        image = imageio.imread(im_file)
        image = resize(image, TARGET_SHAPE, order=3, preserve_range=True)

        labels = imageio.imread(left_file)
        labels += imageio.imread(right_file)
        labels = resize(labels, TARGET_SHAPE, order=0, anti_aliasing=False, preserve_range=True).astype(labels.dtype)

        fname = os.path.basename(im_file)

        # use 10 percent as test data
        if np.random.rand() > 0.8:
            out_image = out_image_test
            out_label = out_label_test
        else:
            out_image = out_image_train
            out_label = out_label_train

        imageio.imwrite(
            os.path.join(out_image, fname), image
        )
        imageio.imwrite(
            os.path.join(out_label, fname), labels
        )


def prepare_nih(input_folder, output_folder):
    image_files = glob(os.path.join(input_folder, "images", "*.png"))
    image_files.sort()
    label_files = glob(os.path.join(input_folder, "masks", "*.png"))
    label_files.sort()

    assert len(image_files) > 0
    assert len(image_files) == len(label_files)

    out_image_train = os.path.join(output_folder, "images", "train")
    out_image_test = os.path.join(output_folder, "images", "test")
    os.makedirs(out_image_train, exist_ok=True)
    os.makedirs(out_image_test, exist_ok=True)

    out_label_train = os.path.join(output_folder, "labels", "train")
    out_label_test = os.path.join(output_folder, "labels", "test")
    os.makedirs(out_label_train, exist_ok=True)
    os.makedirs(out_label_test, exist_ok=True)

    for im_file, label_file in tqdm(zip(image_files, label_files), total=len(image_files)):
        image = imageio.imread(im_file)
        image = resize(image, TARGET_SHAPE, order=3, preserve_range=True)

        labels = imageio.imread(label_file)
        labels = resize(labels, TARGET_SHAPE, order=0, anti_aliasing=False, preserve_range=True).astype(labels.dtype)

        fname = os.path.basename(im_file)

        # use 10 percent as test data
        if np.random.rand() > 0.8:
            out_image = out_image_test
            out_label = out_label_test
        else:
            out_image = out_image_train
            out_label = out_label_train

        imageio.imwrite(
            os.path.join(out_image, fname), image
        )
        imageio.imwrite(
            os.path.join(out_label, fname), labels
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--data", type=str, help="Path where the dataset already exists for Lung X-Rays")
    args = parser.parse_args()

    input_root = args.input
    output_root = args.data

    input_montgomery = os.path.join(input_root, "data", "Montgomery", "MontgomerySet")
    output_montgomery = os.path.join(output_root, "montgomery_processed")
    os.makedirs(output_montgomery, exist_ok=True)
    prepare_montgomery(input_montgomery, output_montgomery)

    input_nih = os.path.join(input_root, "data", "NIH")
    assert os.path.exists(input_nih)
    output_nih = os.path.join(output_root, "nih_processed")
    os.makedirs(output_nih, exist_ok=True)
    prepare_nih(input_nih, output_nih)

    input_jsrt = os.path.join(input_root, "jsrt/Segmentation01")
    output_jsrt = os.path.join(output_root, "jsrt1")
    copytree(input_jsrt, output_jsrt)

    input_jsrt = os.path.join(input_root, "jsrt/segmentation02/segmentation")
    output_jsrt = os.path.join(output_root, "jsrt2")
    copytree(input_jsrt, output_jsrt)


if __name__ == "__main__":
    main()
