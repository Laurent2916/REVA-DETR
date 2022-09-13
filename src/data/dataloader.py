"""Pytorch Lightning DataModules."""

import albumentations as A
import pytorch_lightning as pl
import wandb
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from .dataset import LabeledDataset, RealDataset


def collate_fn(batch):
    return tuple(zip(*batch))


class Spheres(pl.LightningDataModule):
    """Pytorch Lightning DataModule, encapsulating common PyTorch functions."""

    def train_dataloader(self) -> DataLoader:
        """PyTorch training Dataloader.

        Returns:
            DataLoader: the training dataloader
        """
        transforms = A.Compose(
            [
                # A.Flip(),
                # A.ColorJitter(),
                # A.ToGray(p=0.01),
                # A.GaussianBlur(),
                # A.MotionBlur(),
                # A.ISONoise(),
                # A.ImageCompression(),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255,
                ),  # [0, 255] -> coco (?) normalized
                ToTensorV2(),  # HWC -> CHW
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                min_area=0.0,
                min_visibility=0.0,
                label_fields=["labels"],
            ),
        )

        # dataset = LabeledDataset(image_dir="/dev/shm/TRAIN/", transforms=transforms)
        dataset = LabeledDataset(image_dir=wandb.config.DIR_TRAIN_IMG, transforms=transforms)
        # dataset = Subset(dataset, range(6 * 200))  # subset for debugging purpose
        # dataset = Subset(dataset, [0] * 320)  # overfit test

        return DataLoader(
            dataset,
            shuffle=True,
            persistent_workers=True,
            prefetch_factor=wandb.config.PREFETCH_FACTOR,
            batch_size=wandb.config.TRAIN_BATCH_SIZE,
            pin_memory=wandb.config.PIN_MEMORY,
            num_workers=wandb.config.WORKERS,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """PyTorch validation Dataloader.

        Returns:
            DataLoader: the validation dataloader
        """
        transforms = A.Compose(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255,
                ),  # [0, 255] -> [0.0, 1.0] normalized
                ToTensorV2(),  # HWC -> CHW
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                min_area=0.0,
                min_visibility=0.0,
                label_fields=["labels"],
            ),
        )

        # dataset = RealDataset(root="/dev/shm/TEST/", transforms=transforms)
        dataset = RealDataset(root=wandb.config.DIR_VALID_IMG, transforms=transforms)

        return DataLoader(
            dataset,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=wandb.config.PREFETCH_FACTOR,
            batch_size=wandb.config.VALID_BATCH_SIZE,
            pin_memory=wandb.config.PIN_MEMORY,
            num_workers=wandb.config.WORKERS,
            collate_fn=collate_fn,
        )
