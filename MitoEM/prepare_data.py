import os
import argparse

from torch_em.data.datasets.util import download_source, unzip

def download_lucchi_data(_path, download=True):

    path = _path + "lucchi/"

    URL = "http://www.casser.io/files/lucchi_pp.zip"
    CHECKSUM = "770ce9e98fc6f29c1b1a250c637e6c5125f2b5f1260e5a7687b55a79e2e8844d"

    # download and unzip the data
    if os.path.exists(path):
        return path

    os.makedirs(path)
    tmp_path = os.path.join(path, "lucchi.zip")
    download_source(tmp_path, URL, download, checksum=CHECKSUM)
    unzip(tmp_path, path, remove=True)

    root = os.path.join(path, "Lucchi++")
    assert os.path.exists(root), root


def download_vnc_data(_path, download=True):

    path = _path + "vnc/"

    URL = "https://github.com/unidesigner/groundtruth-drosophila-vnc/archive/refs/heads/master.zip"
    CHECKSUM = "f7bd0db03c86b64440a16b60360ad60c0a4411f89e2c021c7ee2c8d6af3d7e86"

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "vnc.zip")
    download_source(zip_path, URL, download, CHECKSUM)
    unzip(zip_path, path, remove=True)

    root = os.path.join(path, "groundtruth-drosophila-vnc-master")
    assert os.path.exists(root)
    

def main(args):
    download_lucchi_data(_path = args.data)
    download_vnc_data(_path = args.data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="~/data/", help="Path where the dataset will be downloaded by the functions")
    args = parser.parse_args()
    main(args)
