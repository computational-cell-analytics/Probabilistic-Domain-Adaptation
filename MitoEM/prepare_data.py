import os
import argparse

from torch_em.data.datasets.util import download_source, unzip


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


def main(args):
    download_lucchi_data(path=args.data)
    download_vnc_data(path=args.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="/mnt/lustre-grete/usr/u16934/data",
        help="Path where the dataset will be downloaded by the functions"
    )
    args = parser.parse_args()
    main(args)
