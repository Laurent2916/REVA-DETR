import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Subset

import wandb

from .dataset import RealDataset


def collate_fn(batch):
    return tuple(zip(*batch))


class Spheres(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def train_dataloader(self):
        transforms = A.Compose(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255,
                ),  # [0, 255] -> [0.0, 1.0] normalized
                # A.ToFloat(max_value=255),
                ToTensorV2(),  # HWC -> CHW
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                min_area=0.0,
                min_visibility=0.0,
                label_fields=["labels"],
            ),
        )

        dataset = RealDataset(root="/dev/shm/TEST_tmp_mrcnn/", transforms=transforms)
        dataset = Subset(dataset, list(range(len(dataset))))  # somehow this allows to better utilize the gpu
        # dataset = Subset(dataset, list(range(20)))  # somehow this allows to better utilize the gpu

        return DataLoader(
            dataset,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=wandb.config.PREFETCH_FACTOR,
            batch_size=wandb.config.TRAIN_BATCH_SIZE,
            pin_memory=wandb.config.PIN_MEMORY,
            num_workers=wandb.config.WORKERS,
            collate_fn=collate_fn,
        )

    # def val_dataloader(self):
    #     dataset = LabeledDataset(image_dir=wandb.config.DIR_VALID_IMG)
    #     dataset = Subset(dataset, list(range(len(dataset))))  # somehow this allows to better utilize the gpu

    #     return DataLoader(
    #         dataset,
    #         shuffle=False,
    #         prefetch_factor=wandb.config.PREFETCH_FACTOR,
    #         batch_size=wandb.config.VAL_BATCH_SIZE,
    #         num_workers=wandb.config.WORKERS,
    #         pin_memory=wandb.config.PIN_MEMORY,
    #     )
