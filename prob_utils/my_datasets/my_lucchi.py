import os
import h5py
import imageio
import numpy as np
from tqdm import tqdm
from glob import glob
from shutil import rmtree
from concurrent import futures

from torch_em.data.datasets.util import download_source, unzip

from .my_segmentation_datasets import default_dual_segmentation_loader

URL = "http://www.casser.io/files/lucchi_pp.zip"
CHECKSUM = "770ce9e98fc6f29c1b1a250c637e6c5125f2b5f1260e5a7687b55a79e2e8844d"

# data from: https://sites.google.com/view/connectomics/
# TODO: add sampler for foreground to avoid empty batches


def _load_volume(path, pattern):
    nz = len(glob(os.path.join(path, "*.png")))
    im0 = imageio.imread(os.path.join(path, pattern % 0))
    out = np.zeros((nz,) + im0.shape, dtype=im0.dtype)
    out[0] = im0

    def _loadz(z):
        im = imageio.imread(os.path.join(path, pattern % z))
        out[z] = im

    n_threads = 8
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(
            tp.map(_loadz, range(1, nz)), desc="Load volume", total=nz-1
        ))

    return out


def _create_data(root, inputs, out_path):
    raw = _load_volume(os.path.join(root, inputs[0]), pattern="mask%04i.png")
    labels_argb = _load_volume(os.path.join(root, inputs[1]), pattern="%i.png")
    if labels_argb.ndim == 4:
        labels = np.zeros(raw.shape, dtype="uint8")
        fg_mask = (labels_argb == np.array([255, 255, 255, 255])[None, None, None]).all(axis=-1)
        labels[fg_mask] = 1
    else:
        assert labels_argb.ndim == 3
        labels = labels_argb
        labels[labels == 255] = 1
    assert (np.unique(labels) == np.array([0, 1])).all()
    assert raw.shape == labels.shape, f"{raw.shape}, {labels.shape}"
    with h5py.File(out_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels", data=labels.astype("uint8"), compression="gzip")


def _require_lucchi_data(path, download):
    # download and unzip the data
    if os.path.exists(path):
        return path

    os.makedirs(path)
    tmp_path = os.path.join(path, "lucchi.zip")
    download_source(tmp_path, URL, download, checksum=CHECKSUM)
    unzip(tmp_path, path, remove=True)

    root = os.path.join(path, "Lucchi++")
    assert os.path.exists(root), root

    inputs = [["Test_In", "Test_Out"], ["Train_In", "Train_Out"]]
    outputs = ["lucchi_train.h5", "lucchi_test.h5"]
    for inp, out in zip(inputs, outputs):
        out_path = os.path.join(path, out)
        _create_data(root, inp, out_path)

    rmtree(root)


def get_lucchi_loader(path, split, download=False, ndim=3, **kwargs):
    assert split in ("train", "test")
    _require_lucchi_data(path, download)
    data_path = os.path.join(path, f"lucchi_{split}.h5")
    assert os.path.exists(data_path), data_path
    raw_key, label_key = "raw", "labels"
    return default_dual_segmentation_loader(
        data_path, raw_key, data_path, label_key, ndim=ndim, **kwargs
    )