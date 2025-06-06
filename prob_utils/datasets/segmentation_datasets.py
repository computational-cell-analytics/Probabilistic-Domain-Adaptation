import os
import warnings
from glob import glob
from copy import deepcopy

import numpy as np

from elf.io import open_file
from elf.wrapper import RoiWrapper

import torch

from torch_em.transform import get_raw_transform
from torch_em.util import ensure_spatial_array, ensure_tensor_with_channels
from torch_em.segmentation import (
    ConcatDataset, samples_to_datasets, check_paths, is_segmentation_dataset, _get_default_transform
)

from prob_utils.my_datasets.my_image_collection_dataset import DualImageCollectionDataset


class DualSegmentationDataset(torch.utils.data.Dataset):
    """
    """
    max_sampling_attempts = 500

    @staticmethod
    def compute_len(shape, patch_shape):
        n_samples = int(np.prod([float(sh / csh) for sh, csh in zip(shape, patch_shape)]))
        return n_samples

    def __init__(
        self,
        raw_path,
        raw_key,
        label_path,
        label_key,
        patch_shape,
        raw_transform=None,
        label_transform=None,
        label_transform2=None,
        augmentation1=None,
        augmentation2=None,
        transform=None,
        roi=None,
        dtype=torch.float32,
        label_dtype=torch.float32,
        n_samples=None,
        sampler=None,
        ndim=None,
        with_channels=False,
        with_label_channels=False,
    ):
        self.raw_path = raw_path
        self.raw_key = raw_key
        self.raw = open_file(raw_path, mode="r")[raw_key]

        self.label_path = label_path
        self.label_key = label_key
        self.labels = open_file(label_path, mode="r")[label_key]

        self._with_channels = with_channels
        self._with_label_channels = with_label_channels

        if roi is not None:
            if isinstance(roi, slice):
                roi = (roi,)
            self.raw = RoiWrapper(self.raw, (slice(None),) + roi) if self._with_channels else RoiWrapper(self.raw, roi)
            self.labels = RoiWrapper(self.labels, (slice(None),) + roi) if self._with_label_channels else\
                RoiWrapper(self.labels, roi)

        shape_raw = self.raw.shape[1:] if self._with_channels else self.raw.shape
        shape_label = self.labels.shape[1:] if self._with_label_channels else self.labels.shape
        assert shape_raw == shape_label, f"{shape_raw}, {shape_label}"

        self.shape = shape_raw
        self.roi = roi

        self._ndim = len(shape_raw) if ndim is None else ndim
        assert self._ndim in (2, 3, 4), f"Invalid data dimensions: {self._ndim}. Only 2d, 3d or 4d data is supported"
        assert len(patch_shape) in (self._ndim, self._ndim + 1), f"{patch_shape}, {self._ndim}"
        self.patch_shape = patch_shape

        self.raw_transform = raw_transform
        self.label_transform = label_transform
        self.label_transform2 = label_transform2
        self.transform = transform
        self.sampler = sampler

        self.augmentation1 = augmentation1
        self.augmentation2 = augmentation2

        self.dtype = dtype
        self.label_dtype = label_dtype

        self._len = self.compute_len(self.shape, self.patch_shape) if n_samples is None else n_samples

        self.sample_shape = patch_shape
        self.trafo_halo = None
        # TODO add support for trafo halo: asking for a bigger bounding box before applying the trafo,
        # which is then cut. See code below; but this ne needs to be properly tested

        # self.trafo_halo = None if self.transform is None else self.transform.halo(self.patch_shape)
        # if self.trafo_halo is not None:
        #     if len(self.trafo_halo) == 2 and self._ndim == 3:
        #         self.trafo_halo = (0,) + self.trafo_halo
        #     assert len(self.trafo_halo) == self._ndim
        #     self.sample_shape = tuple(sh + ha for sh, ha in zip(self.patch_shape, self.trafo_halo))
        #     self.inner_bb = tuple(slice(ha, sh - ha) for sh, ha in zip(self.patch_shape, self.trafo_halo))

    def __len__(self):
        return self._len

    @property
    def ndim(self):
        return self._ndim

    def _sample_bounding_box(self):
        bb_start = [
            np.random.randint(0, sh - psh) if sh - psh > 0 else 0
            for sh, psh in zip(self.shape, self.sample_shape)
        ]
        return tuple(slice(start, start + psh) for start, psh in zip(bb_start, self.sample_shape))

    def _get_sample(self, index):
        if self.raw is None or self.labels is None:
            raise RuntimeError("SegmentationDataset has not been properly deserialized.")
        bb = self._sample_bounding_box()
        bb_raw = (slice(None),) + bb if self._with_channels else bb
        bb_labels = (slice(None),) + bb if self._with_label_channels else bb
        raw, labels = self.raw[bb_raw], self.labels[bb_labels]

        if self.sampler is not None:
            sample_id = 0
            while not self.sampler(raw, labels):
                bb = self._sample_bounding_box()
                bb_raw = (slice(None),) + bb if self._with_channels else bb
                bb_labels = (slice(None),) + bb if self._with_label_channels else bb
                raw, labels = self.raw[bb_raw], self.labels[bb_labels]
                sample_id += 1
                if sample_id > self.max_sampling_attempts:
                    raise RuntimeError(f"Could not sample a valid batch in {self.max_sampling_attempts} attempts")

        # squeeze the singleton spatial axis if we have a spatial shape that is larger by one than self._ndim
        if len(self.patch_shape) == self._ndim + 1:
            raw = raw.squeeze(1 if self._with_channels else 0)
            labels = labels.squeeze(1 if self._with_label_channels else 0)

        return raw, labels

    def crop(self, tensor):
        bb = self.inner_bb
        if tensor.ndim > len(bb):
            bb = (tensor.ndim - len(bb)) * (slice(None),) + bb
        return tensor[bb]

    def __getitem__(self, index):
        raw, labels = self._get_sample(index)
        initial_label_dtype = labels.dtype

        if self.raw_transform is not None:
            raw = self.raw_transform(raw)

        if self.label_transform is not None:
            labels = self.label_transform(labels)

        if self.transform is not None:
            raw, labels = self.transform(raw, labels)
            if self.trafo_halo is not None:
                raw = self.crop(raw)
                labels = self.crop(labels)

        raw1, raw2 = deepcopy(raw), deepcopy(raw)

        if self.augmentation1 is not None:
            raw1 = self.augmentation1(raw1)

        if self.augmentation2 is not None:
            raw2 = self.augmentation2(raw2)

        # support enlarging bounding box here as well (for affinity transform) ?
        if self.label_transform2 is not None:
            labels = ensure_spatial_array(labels, self.ndim, dtype=initial_label_dtype)
            labels = self.label_transform2(labels)

        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)
        raw1 = ensure_tensor_with_channels(raw1, ndim=self._ndim, dtype=self.dtype)
        raw2 = ensure_tensor_with_channels(raw2, ndim=self._ndim, dtype=self.dtype)
        labels = ensure_tensor_with_channels(labels, ndim=self._ndim, dtype=self.label_dtype)

        if self.augmentation1 or self.augmentation2 is not None:
            return raw, raw1, raw2, labels
        else:
            return raw, labels

    # need to overwrite pickle to support h5py
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["raw"]
        del state["labels"]
        return state

    def __setstate__(self, state):
        raw_path, raw_key = state["raw_path"], state["raw_key"]
        label_path, label_key = state["label_path"], state["label_key"]
        roi = state["roi"]
        try:
            raw = open_file(raw_path, mode="r")[raw_key]
            if roi is not None:
                raw = RoiWrapper(raw, (slice(None),) + roi) if state["_with_channels"] else RoiWrapper(raw, roi)
            state["raw"] = raw
        except Exception:
            msg = f"SegmentationDataset could not be deserialized because of missing {raw_path}, {raw_key}.\n"
            msg += "The dataset is deserialized in order to allow loading trained models from a checkpoint.\n"
            msg += "But it cannot be used for further training and wil throw an error."
            warnings.warn(msg)
            state["raw"] = None

        try:
            labels = open_file(label_path, mode="r")[label_key]
            if roi is not None:
                labels = RoiWrapper(labels, (slice(None),) + roi) if state["_with_label_channels"] else\
                    RoiWrapper(labels, roi)
            state["labels"] = labels
        except Exception:
            msg = f"SegmentationDataset could not be deserialized because of missing {label_path}, {label_key}.\n"
            msg += "The dataset is deserialized in order to allow loading trained models from a checkpoint.\n"
            msg += "But it cannot be used for further training and wil throw an error."
            warnings.warn(msg)
            state["labels"] = None

        self.__dict__.update(state)


def _load_dual_segmentation_dataset(raw_paths, raw_key, label_paths, label_key, **kwargs):
    rois = kwargs.pop("rois", None)
    if isinstance(raw_paths, str):
        if rois is not None:
            assert isinstance(rois, (tuple, slice))
            if isinstance(rois, tuple):
                assert all(isinstance(roi, slice) for roi in rois)
        ds = DualSegmentationDataset(raw_paths, raw_key, label_paths, label_key, roi=rois, **kwargs)
    else:
        assert len(raw_paths) > 0
        if rois is not None:
            assert len(rois) == len(label_paths)
            assert all(isinstance(roi, tuple) for roi in rois)
        n_samples = kwargs.pop("n_samples", None)

        samples_per_ds = (
            [None] * len(raw_paths) if n_samples is None else samples_to_datasets(n_samples, raw_paths, raw_key)
        )
        ds = []
        for i, (raw_path, label_path) in enumerate(zip(raw_paths, label_paths)):
            roi = None if rois is None else rois[i]
            dset = DualSegmentationDataset(
                raw_path, raw_key, label_path, label_key, roi=roi, n_samples=samples_per_ds[i], **kwargs
            )
            ds.append(dset)
        ds = ConcatDataset(*ds)
    return ds


def _load_image_collection_dataset(raw_paths, raw_key, label_paths, label_key, roi, **kwargs):
    def _get_paths(rpath, rkey, lpath, lkey, this_roi):
        rpath = glob(os.path.join(rpath, rkey))
        rpath.sort()
        if len(rpath) == 0:
            raise ValueError(f"Could not find any images for pattern {os.path.join(rpath, rkey)}")
        lpath = glob(os.path.join(lpath, lkey))
        lpath.sort()
        if len(rpath) != len(lpath):
            raise ValueError(f"Expect same number of raw and label images, got {len(rpath)}, {len(lpath)}")

        if this_roi is not None:
            rpath, lpath = rpath[roi], lpath[roi]

        return rpath, lpath

    patch_shape = kwargs.pop("patch_shape")
    if len(patch_shape) == 3:
        if patch_shape[0] != 1:
            raise ValueError(f"Image collection dataset expects 2d patch shape, got {patch_shape}")
        patch_shape = patch_shape[1:]
    assert len(patch_shape) == 2

    if isinstance(raw_paths, str):
        raw_paths, label_paths = _get_paths(raw_paths, raw_key, label_paths, label_key, roi)
        ds = DualImageCollectionDataset(raw_paths, label_paths, patch_shape=patch_shape, **kwargs)
    elif raw_key is None:
        assert label_key is None
        assert isinstance(raw_paths, (list, tuple)) and isinstance(label_paths, (list, tuple))
        assert len(raw_paths) == len(label_paths)
        ds = DualImageCollectionDataset(raw_paths, label_paths, patch_shape=patch_shape, **kwargs)
    else:
        ds = []
        n_samples = kwargs.pop("n_samples", None)
        samples_per_ds = (
            [None] * len(raw_paths) if n_samples is None else samples_to_datasets(n_samples, raw_paths, raw_key)
        )
        if roi is None:
            roi = len(raw_paths) * [None]
        assert len(roi) == len(raw_paths)
        for i, (raw_path, label_path, this_roi) in enumerate(zip(raw_paths, label_paths, roi)):
            rpath, lpath = _get_paths(raw_path, raw_key, label_path, label_key, this_roi)
            dset = DualImageCollectionDataset(
                rpath, lpath, patch_shape=patch_shape, n_samples=samples_per_ds[i], **kwargs
            )
            ds.append(dset)
        ds = ConcatDataset(*ds)
    return ds


def default_dual_segmentation_dataset(
    raw_paths,
    raw_key,
    label_paths,
    label_key,
    patch_shape,
    label_transform=None,
    label_transform2=None,
    augmentation1=None,
    augmentation2=None,
    raw_transform=None,
    transform=None,
    dtype=torch.float32,
    label_dtype=torch.float32,
    rois=None,
    n_samples=None,
    sampler=None,
    ndim=None,
    is_seg_dataset=None,
    with_channels=False,
    with_label_channels=False,
):
    check_paths(raw_paths, label_paths)

    # we add the train labels to test in order to just pass it to the trainer
    # (we are not using the labels, so for now it is not an issue for us)
    if type(label_paths) is not list:
        if label_paths.split("/")[-1][:-3] == "vnc_test":
            label_paths = label_paths[:-7] + "train.h5"

    if is_seg_dataset is None:
        is_seg_dataset = is_segmentation_dataset(raw_paths, raw_key, label_paths, label_key)

    # we always use a raw transform in the convenience function
    if raw_transform is None:
        raw_transform = get_raw_transform()

    # we always use augmentations in the convenience function
    if transform is None:
        transform = _get_default_transform(
            raw_paths if isinstance(raw_paths, str) else raw_paths[0], raw_key, is_seg_dataset, ndim
        )

    if is_seg_dataset:
        ds = _load_dual_segmentation_dataset(
            raw_paths,
            raw_key,
            label_paths,
            label_key,
            patch_shape=patch_shape,
            raw_transform=raw_transform,
            label_transform=label_transform,
            label_transform2=label_transform2,
            augmentation1=augmentation1,
            augmentation2=augmentation2,
            transform=transform,
            rois=rois,
            n_samples=n_samples,
            sampler=sampler,
            ndim=ndim,
            dtype=dtype,
            label_dtype=label_dtype,
            with_channels=with_channels,
            with_label_channels=with_label_channels,
        )
    else:
        print("Not our use case for now, please have a look")
        quit()
        ds = _load_image_collection_dataset(
            raw_paths,
            raw_key,
            label_paths,
            label_key,
            roi=rois,
            patch_shape=patch_shape,
            label_transform=label_transform,
            raw_transform=raw_transform,
            label_transform2=label_transform2,
            transform=transform,
            n_samples=n_samples,
            dtype=dtype,
            label_dtype=label_dtype,
        )

    return ds


def default_dual_segmentation_loader(
    raw_paths,
    raw_key,
    label_paths,
    label_key,
    batch_size,
    patch_shape,
    label_transform=None,
    label_transform2=None,
    augmentation1=None,
    augmentation2=None,
    raw_transform=None,
    transform=None,
    dtype=torch.float32,
    label_dtype=torch.float32,
    rois=None,
    n_samples=None,
    sampler=None,
    ndim=None,
    is_seg_dataset=None,
    with_channels=False,
    with_label_channels=False,
    **loader_kwargs,
):
    ds = default_dual_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=raw_key,
        label_paths=label_paths,
        label_key=label_key,
        patch_shape=patch_shape,
        label_transform=label_transform,
        label_transform2=label_transform2,
        augmentation1=augmentation1,
        augmentation2=augmentation2,
        raw_transform=raw_transform,
        transform=transform,
        dtype=dtype,
        label_dtype=label_dtype,
        rois=rois,
        n_samples=n_samples,
        sampler=sampler,
        ndim=ndim,
        is_seg_dataset=is_seg_dataset,
        with_channels=with_channels,
        with_label_channels=with_label_channels,
    )
    return get_dual_data_loader(ds, batch_size=batch_size, **loader_kwargs)


def get_dual_data_loader(dataset: torch.utils.data.Dataset, batch_size, **loader_kwargs) -> torch.utils.data.DataLoader:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, **loader_kwargs)
    # monkey patch shuffle attribute to the loader
    loader.shuffle = loader_kwargs.get("shuffle", False)
    return loader
