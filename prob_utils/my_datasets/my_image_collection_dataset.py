import numpy as np
from copy import deepcopy

import torch
from torch_em.util import (ensure_spatial_array, ensure_tensor_with_channels,
                    load_image, supports_memmap, ensure_tensor)


# TODO pad images that are too small for the patch shape
class ImageCollectionDataset(torch.utils.data.Dataset):
    max_sampling_attempts = 500

    def _check_inputs(self, raw_images, label_images):
        if len(raw_images) != len(label_images):
            raise ValueError(f"Expect same number of  and label images, got {len(raw_images)} and {len(label_images)}")

        is_multichan = None
        for raw_im, label_im in zip(raw_images, label_images):

            # we only check for compatible shapes if both images support memmap, because
            # we don't want to load everything into ram
            if supports_memmap(raw_im) and supports_memmap(label_im):
                shape = load_image(raw_im).shape
                assert len(shape) in (2, 3)

                multichan = len(shape) == 3
                if is_multichan is None:
                    is_multichan = multichan
                else:
                    assert is_multichan == multichan

                # we assume axis last
                if is_multichan:
                    # use heuristic to decide whether the data is stored in channel last or channel first order:
                    # if the last axis has a length smaller than 16 we assume that it's the channel axis,
                    # otherwise we assume it's a spatial axis and that the first axis is the channel axis.
                    if shape[-1] < 16:
                        shape = shape[:-1]
                    else:
                        shape = shape[1:]

                label_shape = load_image(label_im).shape
                if shape != label_shape:
                    msg = f"Expect raw and labels of same shape, got {shape}, {label_shape} for {raw_im}, {label_im}"
                    raise ValueError(msg)

    def __init__(
        self,
        raw_image_paths,
        label_image_paths,
        consensus_mask_paths,
        patch_shape,
        raw_transform=None,
        label_transform=None,
        label_transform2=None,
        transform=None,
        dtype=torch.float32,
        label_dtype=torch.float32,
        n_samples=None,
        sampler=None,
    ):
        self._check_inputs(raw_image_paths, label_image_paths)
        self.raw_images = raw_image_paths
        self.label_images = label_image_paths
        self.consensus_masks = consensus_mask_paths
        self._ndim = 2

        assert len(patch_shape) == self._ndim
        self.patch_shape = patch_shape

        self.raw_transform = raw_transform
        self.label_transform = label_transform
        self.label_transform2 = label_transform2
        self.transform = transform
        self.sampler = sampler

        self.dtype = dtype
        self.label_dtype = label_dtype

        if n_samples is None:
            self._len = len(self.raw_images)
            self.sample_random_index = False
        else:
            self._len = n_samples
            self.sample_random_index = True

    def __len__(self):
        return self._len

    @property
    def ndim(self):
        return self._ndim

    def _sample_bounding_box(self, shape):
        if any(sh < psh for sh, psh in zip(shape, self.patch_shape)):
            raise NotImplementedError(
                "Image padding is not supported yet. Data shape {shape}, patch shape {self.patch_shape}"
            )
        bb_start = [
            np.random.randint(0, sh - psh) if sh - psh > 0 else 0
            for sh, psh in zip(shape, self.patch_shape)
        ]
        return tuple(slice(start, start + psh) for start, psh in zip(bb_start, self.patch_shape))

    def _get_sample(self, index):
        if self.sample_random_index:
            index = np.random.randint(0, len(self.raw_images))
        # these are just the file paths
        raw, label, consensus = self.raw_images[index], self.label_images[index], self.consensus_masks[index]
        raw = load_image(raw)
        label = load_image(label)
        consensus = load_image(consensus)

        have_raw_channels = raw.ndim == 3
        have_label_channels = label.ndim == 3
        if have_label_channels:
            raise NotImplementedError("Multi-channel labels are not supported.")

        shape = raw.shape
        # we determine if image has channels as te first or last axis base on array shape.
        # This will work only for images with less than 16 channels.
        prefix_box = tuple()
        if have_raw_channels:
            # use heuristic to decide whether the data is stored in channel last or channel first order:
            # if the last axis has a length smaller than 16 we assume that it's the channel axis,
            # otherwise we assume it's a spatial axis and that the first axis is the channel axis.
            if shape[-1] < 16:
                shape = shape[:-1]
            else:
                shape = shape[1:]
                prefix_box = (slice(None), )

        # sample random bounding box for this image
        bb = self._sample_bounding_box(shape)
        raw_patch = np.array(raw[prefix_box + bb])
        label_patch = np.array(label[bb])
        consensus_patch = np.array(consensus[bb])

        if self.sampler is not None:
            sample_id = 0
            while not self.sampler(raw_patch, label_patch):
                bb = self._sample_bounding_box(shape)
                raw_patch = np.array(raw[prefix_box + bb])
                label_patch = np.array(label[bb])
                sample_id += 1
                if sample_id > self.max_sampling_attempts:
                    raise RuntimeError(f"Could not sample a valid batch in {self.max_sampling_attempts} attempts")

        # to channel first
        if have_raw_channels and len(prefix_box) == 0:
            raw_patch = raw_patch.transpose((2, 0, 1))

        return raw_patch, label_patch, consensus_patch

    def __getitem__(self, index):
        raw, labels, consensus = self._get_sample(index)
        initial_label_dtype = labels.dtype

        if self.raw_transform is not None:
            raw = self.raw_transform(raw)

        if self.label_transform is not None:
            labels = self.label_transform(labels)
            consensus = self.label_transform(labels)

        if self.transform is not None:
            raw, labels, consensus = self.transform(raw, labels, consensus)
            # if self.trafo_halo is not None:
            #     raw = self.crop(raw)
            #     labels = self.crop(labels)

        # support enlarging bounding box here as well (for affinity transform) ?
        if self.label_transform2 is not None:
            labels = ensure_spatial_array(labels, self.ndim, dtype=initial_label_dtype)
            labels = self.label_transform2(labels)

        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)
        labels = ensure_tensor_with_channels(labels, ndim=self._ndim, dtype=self.label_dtype)
        consensus = ensure_tensor_with_channels(consensus, ndim=self._ndim, dtype=torch.int32)
        return raw, labels, consensus

# TODO pad images that are too small for the patch shape
class DualImageCollectionDataset(torch.utils.data.Dataset):
    max_sampling_attempts = 500

    def _check_inputs(self, raw_images, label_images):
        if len(raw_images) != len(label_images):
            raise ValueError(f"Expect same number of  and label images, got {len(raw_images)} and {len(label_images)}")

        is_multichan = None
        for raw_im, label_im in zip(raw_images, label_images):

            # we only check for compatible shapes if both images support memmap, because
            # we don't want to load everything into ram
            if supports_memmap(raw_im) and supports_memmap(label_im):
                shape = load_image(raw_im).shape
                assert len(shape) in (2, 3)

                multichan = len(shape) == 3
                if is_multichan is None:
                    is_multichan = multichan
                else:
                    assert is_multichan == multichan

                # we assume axis last
                if is_multichan:
                    # use heuristic to decide whether the data is stored in channel last or channel first order:
                    # if the last axis has a length smaller than 16 we assume that it's the channel axis,
                    # otherwise we assume it's a spatial axis and that the first axis is the channel axis.
                    if shape[-1] < 16:
                        shape = shape[:-1]
                    else:
                        shape = shape[1:]

                label_shape = load_image(label_im).shape
                if shape != label_shape:
                    msg = f"Expect raw and labels of same shape, got {shape}, {label_shape} for {raw_im}, {label_im}"
                    raise ValueError(msg)

    def __init__(
        self,
        raw_image_paths,
        label_image_paths,
        patch_shape,
        raw_transform=None,
        label_transform=None,
        label_transform2=None,
        augmentation1=None,
        augmentation2=None,
        transform=None,
        dtype=torch.float32,
        label_dtype=torch.float32,
        n_samples=None,
        sampler=None,
    ):
        self._check_inputs(raw_image_paths, label_image_paths)
        self.raw_images = raw_image_paths
        self.label_images = label_image_paths
        self._ndim = 2

        assert len(patch_shape) == self._ndim
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

        if n_samples is None:
            self._len = len(self.raw_images)
            self.sample_random_index = False
        else:
            self._len = n_samples
            self.sample_random_index = True

    def __len__(self):
        return self._len

    @property
    def ndim(self):
        return self._ndim

    def _sample_bounding_box(self, shape):
        if any(sh < psh for sh, psh in zip(shape, self.patch_shape)):
            raise NotImplementedError(
                "Image padding is not supported yet. Data shape {shape}, patch shape {self.patch_shape}"
            )
        bb_start = [
            np.random.randint(0, sh - psh) if sh - psh > 0 else 0
            for sh, psh in zip(shape, self.patch_shape)
        ]
        return tuple(slice(start, start + psh) for start, psh in zip(bb_start, self.patch_shape))

    def _get_sample(self, index):
        if self.sample_random_index:
            index = np.random.randint(0, len(self.raw_images))
        # these are just the file paths
        raw, label = self.raw_images[index], self.label_images[index]
        raw = load_image(raw)
        label = load_image(label)

        have_raw_channels = raw.ndim == 3
        have_label_channels = label.ndim == 3
        if have_label_channels:
            raise NotImplementedError("Multi-channel labels are not supported.")

        shape = raw.shape
        # we determine if image has channels as te first or last axis base on array shape.
        # This will work only for images with less than 16 channels.
        prefix_box = tuple()
        if have_raw_channels:
            # use heuristic to decide whether the data is stored in channel last or channel first order:
            # if the last axis has a length smaller than 16 we assume that it's the channel axis,
            # otherwise we assume it's a spatial axis and that the first axis is the channel axis.
            if shape[-1] < 16:
                shape = shape[:-1]
            else:
                shape = shape[1:]
                prefix_box = (slice(None), )

        # sample random bounding box for this image
        bb = self._sample_bounding_box(shape)
        raw_patch = np.array(raw[prefix_box + bb])
        label_patch = np.array(label[bb])

        if self.sampler is not None:
            sample_id = 0
            while not self.sampler(raw_patch, label_patch):
                bb = self._sample_bounding_box(shape)
                raw_patch = np.array(raw[prefix_box + bb])
                label_patch = np.array(label[bb])
                sample_id += 1
                if sample_id > self.max_sampling_attempts:
                    raise RuntimeError(f"Could not sample a valid batch in {self.max_sampling_attempts} attempts")

        # to channel first
        if have_raw_channels and len(prefix_box) == 0:
            raw_patch = raw_patch.transpose((2, 0, 1))

        return raw_patch, label_patch

    def __getitem__(self, index):
        raw, labels = self._get_sample(index)
        initial_label_dtype = labels.dtype

        if self.raw_transform is not None:
            raw = self.raw_transform(raw)

        if self.label_transform is not None:
            labels = self.label_transform(labels)

        if self.transform is not None:
            raw, labels = self.transform(raw, labels)
            # if self.trafo_halo is not None:
            #     raw = self.crop(raw)
            #     labels = self.crop(labels)
        
        raw1, raw2 = deepcopy(raw), deepcopy(raw)
        raw1, raw2 = ensure_tensor_with_channels(raw1, ndim=self._ndim, dtype=self.dtype), ensure_tensor_with_channels(raw2, ndim=self._ndim, dtype=self.dtype)

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
