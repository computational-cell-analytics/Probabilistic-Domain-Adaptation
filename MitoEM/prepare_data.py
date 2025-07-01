import os
import argparse

import numpy as np
import imageio.v3 as imageio

from torch_em.data.datasets.util import download_source, unzip

from elf.io import open_file


def download_lucchi_data(path, download=True):

    fpath = os.path.join(path, "lucchi")

    URL = "http://www.casser.io/files/lucchi_pp.zip"
    CHECKSUM = "770ce9e98fc6f29c1b1a250c637e6c5125f2b5f1260e5a7687b55a79e2e8844d"

    # Download and unzip the data
    if os.path.exists(fpath):
        return fpath

    os.makedirs(fpath)
    tmp_path = os.path.join(fpath, "lucchi.zip")
    download_source(tmp_path, URL, download, checksum=CHECKSUM)
    unzip(tmp_path, fpath, remove=True)

    root = os.path.join(fpath, "Lucchi++")
    assert os.path.exists(root), root


def download_vnc_data(path, download=True):

    fpath = os.path.join(path, "vnc")

    URL = "https://github.com/unidesigner/groundtruth-drosophila-vnc/archive/refs/heads/master.zip"
    CHECKSUM = "f7bd0db03c86b64440a16b60360ad60c0a4411f89e2c021c7ee2c8d6af3d7e86"

    os.makedirs(fpath, exist_ok=True)
    zip_path = os.path.join(fpath, "vnc.zip")
    download_source(zip_path, URL, download, CHECKSUM)
    unzip(zip_path, fpath, remove=True)

    root = os.path.join(fpath, "groundtruth-drosophila-vnc-master")
    assert os.path.exists(root)


def download_urocell_data(path, download=True):
    from torch_em.data.datasets.electron_microscopy.uro_cell import get_uro_cell_paths
    paths = get_uro_cell_paths(path=os.path.join(path, "urocell"), target="mito", download=download)

    # NOTE: Use the last volume for testing purpose
    paths = sorted(paths)
    paths = [paths[-1]]

    base_dir = os.path.join(path, "urocell", "preprocessed")
    os.makedirs(base_dir, exist_ok=True)

    # The simplest strategy would be to create slices, since we deal with 2d models.
    for path in paths:
        f = open_file(path, mode="r")
        raw = f["raw"][:]
        labels = f["labels/mito"][:]

        counter = 0
        for r, l in zip(raw, labels):
            if len(np.unique(l)) > 1:  # If there is some foreground, let's use the slices.
                binary_label = (l > 0).astype("uint8")
                imageio.imwrite(os.path.join(base_dir, f"{counter:05}_image.tif"), r, compression="zlib")
                imageio.imwrite(os.path.join(base_dir, f"{counter:05}_gt.tif"), binary_label, compression="zlib")
                counter += 1


def main(args):
    download_lucchi_data(path=args.data)
    download_vnc_data(path=args.data)
    download_urocell_data(path=args.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="/mnt/lustre-grete/usr/u16934/data",
        help="Path where the dataset will be downloaded by the functions"
    )
    args = parser.parse_args()
    main(args)
