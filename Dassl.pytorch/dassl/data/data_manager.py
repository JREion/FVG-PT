import torch
import torchvision.transforms as T
from tabulate import tabulate
from torch.utils.data import Dataset as TorchDataset

from dassl.utils import read_image

from .datasets import build_dataset
from .samplers import build_sampler
from .transforms import INTERPOLATION_MODES, build_transform

import os
import os.path as osp
import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF  # NEW: for geometric transforms (crop/flip)
from torchvision.transforms import InterpolationMode  # NEW: nearest-neighbor interpolation for masks
from tabulate import tabulate
from torch.utils.data import Dataset as TorchDataset


def build_data_loader(
    cfg,
    sampler_type="SequentialSampler",
    data_source=None,
    batch_size=64,
    n_domain=0,
    n_ins=2,
    tfm=None,
    is_train=True,
    dataset_wrapper=None
):
    # Build sampler
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain,
        n_ins=n_ins
    )

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_train and len(data_source) >= batch_size,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
    )
    assert len(data_loader) > 0

    return data_loader


class DataManager:

    def __init__(
        self,
        cfg,
        custom_tfm_train=None,
        custom_tfm_test=None,
        dataset_wrapper=None
    ):
        # Load dataset
        dataset = build_dataset(cfg)

        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # Build train_loader_x
        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        # [FVGPT_Dassl] Build a dataloader with masks
        train_loader_fgpt = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=DatasetWrapperFGPT  # NEW
        )

        # Build train_loader_u
        train_loader_u = None
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
            n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                n_ins=n_ins_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper
            )

        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # [RDBoost_Vis]
        test_loader_fgpt = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=DatasetWrapperFGPT
        )

        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader
        # self.test_loader = test_loader_fgpt  # [RDBoost_Vis]
        self.train_loader_fgpt = train_loader_fgpt  # [FVGPT_Dassl]

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def lab2cname(self):
        return self._lab2cname

    def show_dataset_summary(self, cfg):
        dataset_name = cfg.DATASET.NAME
        source_domains = cfg.DATASET.SOURCE_DOMAINS
        target_domains = cfg.DATASET.TARGET_DOMAINS

        table = []
        table.append(["Dataset", dataset_name])
        if source_domains:
            table.append(["Source", source_domains])
        if target_domains:
            table.append(["Target", target_domains])
        table.append(["# classes", f"{self.num_classes:,}"])
        table.append(["# train_x", f"{len(self.dataset.train_x):,}"])
        if self.dataset.train_u:
            table.append(["# train_u", f"{len(self.dataset.train_u):,}"])
        if self.dataset.val:
            table.append(["# val", f"{len(self.dataset.val):,}"])
        table.append(["# test", f"{len(self.dataset.test):,}"])

        print(tabulate(table))


class DatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform is None".format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "index": idx
        }

        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output["img"] = img
        else:
            output["img"] = img0

        if self.return_img0:
            output["img0"] = self.to_tensor(img0)  # without any augmentation

        return output

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img


class DatasetWrapperFGPT(TorchDataset):
    """Similar to DatasetWrapper, but additionally loads a mask and ensures img and mask share exactly the same geometric transforms.

    Assumptions:
      Example original image path:
        D:\\Datasets\\caltech-101\\caltech-101\\101_ObjectCategories\\accordion\\image_0001.jpg
      Example mask path:
        D:\\Datasets\\caltech-101\\caltech-101\\mask\\101_ObjectCategories\\accordion\\image_0001.jpg

    That is: add a 'mask' directory under DATASET.ROOT; subsequent subpaths match the original images, and filenames are identical.
    """

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        # NOTE:
        #   Keep the exact same semantics as DatasetWrapper: `transform` can be
        #   a callable, or a list/tuple of callables (multi-view). We will apply
        #   geometric transforms to both (img, mask) with shared randomness, while
        #   applying photometric/tensor-only transforms to img only.
        self.transform = transform
        self.is_train = is_train
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform is None".format(self.k_tfm)
            )

        # Same as DatasetWrapper: build a "no-augmentation" to_tensor for img0
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

        self.root = cfg.DATASET.ROOT
        self.mask_root = osp.join(self.root, "mask")

    def __len__(self):
        return len(self.data_source)

    def _get_mask_path(self, impath):
        """Convert the original image path to the corresponding mask path.

        Assume impath looks like:
          ROOT/101_ObjectCategories/accordion/image_0001.jpg
        Then mask_path is:
          ROOT/mask/101_ObjectCategories/accordion/image_0001.jpg
        """
        rel_path = osp.relpath(impath, self.root)
        mask_path = osp.join(self.mask_root, rel_path)
        return mask_path

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "index": idx
        }

        img0 = read_image(item.impath)        # PIL.Image (RGB)
        mask_path = self._get_mask_path(item.impath)
        mask0 = read_image(mask_path)        # PIL.Image (RGB), foreground is white, background is black

        if self.transform is not None:
            # Match DatasetWrapper's behavior: support multi-view (list/tuple)
            # transforms transparently.
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img, mask = self._transform_image_mask(tfm, img0, mask0)
                    k_img = "img" if i == 0 else f"img{i+1}"
                    k_msk = "mask" if i == 0 else f"mask{i+1}"
                    output[k_img] = img
                    output[k_msk] = mask
            else:
                img, mask = self._transform_image_mask(self.transform, img0, mask0)
                output["img"] = img
                output["mask"] = mask
        else:
            # When no data augmentation is used: img uses the raw image, mask only does to_tensor plus binarization
            output["img"] = img0
            mask_t = TF.to_tensor(mask0)
            if mask_t.size(0) == 3:
                mask_t = mask_t.mean(dim=0, keepdim=True)
            output["mask"] = (mask_t > 0.5).float()

        if self.return_img0:
            output["img0"] = self.to_tensor(img0)

        return output

    # -------------------------
    # Paired transform helpers
    # -------------------------
    def _resize_mask_to_img(self, img0, mask0):
        """Resize mask to exactly match img size (keep mask semantic integrity)."""
        if mask0.size != img0.size:
            mask0 = TF.resize(
                mask0,
                [img0.height, img0.width],
                interpolation=InterpolationMode.NEAREST
            )
        return mask0

    def _binarize_mask(self, mask_pil):
        """Convert mask PIL to (1,H,W) float tensor with values in {0,1}."""
        mask_t = TF.to_tensor(mask_pil)
        if mask_t.dim() == 3 and mask_t.size(0) == 3:
            mask_t = mask_t.mean(dim=0, keepdim=True)
        return (mask_t > 0.5).float()

    def _transform_image_mask(self, tfm, img0, mask0):
        """Like DatasetWrapper._transform_image, but for (img, mask) pair."""
        mask0 = self._resize_mask_to_img(img0, mask0)

        img_list, mask_list = [], []
        for _ in range(self.k_tfm):
            img, mask = self._apply_pair_transform(tfm, img0, mask0)

            # Ensure img is tensor if the upstream transform forgot to do so.
            if not isinstance(img, torch.Tensor):
                img = TF.to_tensor(img)

            mask_bin = self._binarize_mask(mask)
            img_list.append(img)
            mask_list.append(mask_bin)

        if len(img_list) == 1:
            return img_list[0], mask_list[0]
        return img_list, mask_list

    def _apply_pair_transform(self, tfm, img, mask):
        """Apply an arbitrary transform pipeline to img, while synchronizing
        *geometric* randomness for mask.

        Strategy:
          - Geometric transforms are applied to both (img, mask) with shared params.
          - Photometric / tensor-only transforms are applied to img only.
          - Unknown transforms are treated as img-only by default.
        """
        if isinstance(tfm, T.Compose):
            for t in tfm.transforms:
                img, mask = self._apply_single_pair(t, img, mask)
            return img, mask

        return self._apply_single_pair(tfm, img, mask)

    def _apply_single_pair(self, t, img, mask):
        """Apply a single transform `t` to (img, mask) pair."""
        # -----------------
        # Container wrappers
        # -----------------
        if isinstance(t, T.RandomApply):
            if random.random() < t.p:
                for tt in t.transforms:
                    img, mask = self._apply_single_pair(tt, img, mask)
            return img, mask

        if isinstance(t, T.RandomOrder):
            order = list(t.transforms)
            random.shuffle(order)
            for tt in order:
                img, mask = self._apply_single_pair(tt, img, mask)
            return img, mask

        if isinstance(t, T.RandomChoice):
            tt = random.choice(t.transforms)
            return self._apply_single_pair(tt, img, mask)

        # -----------------
        # Geometric (paired)
        # -----------------
        if isinstance(t, T.RandomHorizontalFlip):
            if random.random() < t.p:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            return img, mask

        if isinstance(t, T.RandomVerticalFlip):
            if random.random() < t.p:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            return img, mask

        if isinstance(t, T.RandomResizedCrop):
            i, j, h, w = T.RandomResizedCrop.get_params(img, scale=t.scale, ratio=t.ratio)
            img = TF.resized_crop(img, i, j, h, w, t.size, t.interpolation)
            mask = TF.resized_crop(mask, i, j, h, w, t.size, InterpolationMode.NEAREST)
            return img, mask

        if isinstance(t, T.RandomCrop):
            # Follow torchvision's behavior: optional padding and pad_if_needed
            if getattr(t, "padding", None) is not None:
                img = TF.pad(img, t.padding, fill=getattr(t, "fill", 0), padding_mode=t.padding_mode)
                mask = TF.pad(mask, t.padding, fill=0, padding_mode=t.padding_mode)

            if getattr(t, "pad_if_needed", False):
                # Pad left/right
                if img.size[0] < t.size[1]:
                    pad_r = t.size[1] - img.size[0]
                    pad = (0, 0, pad_r, 0)  # (left, top, right, bottom)
                    img = TF.pad(img, pad, fill=getattr(t, "fill", 0), padding_mode=t.padding_mode)
                    mask = TF.pad(mask, pad, fill=0, padding_mode=t.padding_mode)
                # Pad top/bottom
                if img.size[1] < t.size[0]:
                    pad_b = t.size[0] - img.size[1]
                    pad = (0, 0, 0, pad_b)
                    img = TF.pad(img, pad, fill=getattr(t, "fill", 0), padding_mode=t.padding_mode)
                    mask = TF.pad(mask, pad, fill=0, padding_mode=t.padding_mode)

            i, j, h, w = T.RandomCrop.get_params(img, output_size=t.size)
            img = TF.crop(img, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
            return img, mask

        if isinstance(t, T.Resize):
            img = TF.resize(img, t.size, interpolation=t.interpolation)
            mask = TF.resize(mask, t.size, interpolation=InterpolationMode.NEAREST)
            return img, mask

        if isinstance(t, T.CenterCrop):
            img = TF.center_crop(img, t.size)
            mask = TF.center_crop(mask, t.size)
            return img, mask

        if isinstance(t, T.Pad):
            img = TF.pad(img, t.padding, fill=getattr(t, "fill", 0), padding_mode=t.padding_mode)
            mask = TF.pad(mask, t.padding, fill=0, padding_mode=t.padding_mode)
            return img, mask

        if isinstance(t, T.RandomRotation):
            angle = T.RandomRotation.get_params(t.degrees)
            fill_img = getattr(t, "fill", 0)
            img = TF.rotate(img, angle, interpolation=t.interpolation, expand=t.expand, center=t.center, fill=fill_img)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, expand=t.expand, center=t.center, fill=0)
            return img, mask

        if isinstance(t, T.RandomAffine):
            try:
                img_size = TF.get_image_size(img)
            except Exception:
                img_size = img.size
            angle, translations, scale, shear = T.RandomAffine.get_params(
                t.degrees, t.translate, t.scale, t.shear, img_size
            )
            fill_img = getattr(t, "fill", 0)
            img = TF.affine(
                img, angle=angle, translate=translations, scale=scale, shear=shear,
                interpolation=t.interpolation, fill=fill_img, center=t.center
            )
            mask = TF.affine(
                mask, angle=angle, translate=translations, scale=scale, shear=shear,
                interpolation=InterpolationMode.NEAREST, fill=0, center=t.center
            )
            return img, mask

        if isinstance(t, T.RandomPerspective):
            # torchvision has slightly different get_params signatures across versions
            try:
                startpoints, endpoints = t.get_params(img.width, img.height, t.distortion_scale)
            except Exception:
                try:
                    startpoints, endpoints = t.get_params(img.size, t.distortion_scale)
                except Exception:
                    startpoints, endpoints = T.RandomPerspective.get_params(img.width, img.height, t.distortion_scale)

            fill_img = getattr(t, "fill", 0)
            img = TF.perspective(img, startpoints, endpoints, interpolation=t.interpolation, fill=fill_img)
            mask = TF.perspective(mask, startpoints, endpoints, interpolation=InterpolationMode.NEAREST, fill=0)
            return img, mask

        # -------------------------------------------------
        # Photometric / tensor-only transforms (img-only)
        # -------------------------------------------------
        try:
            img = t(img)
        except Exception:
            # If this transform cannot be applied here, keep img unchanged.
            pass
        return img, mask
