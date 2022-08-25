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
                A.ToFloat(max_value=255),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                min_area=0.0,
                min_visibility=0.0,
                label_fields=["labels"],
            ),
        )

        dataset = RealDataset(root="/media/disk1/lfainsin/TEST_tmp_mrcnn/", transforms=transforms)
        print(f"len(dataset)={len(dataset)}")
        dataset = Subset(dataset, list(range(len(dataset))))  # somehow this allows to better utilize the gpu

        return DataLoader(
            dataset,
            shuffle=True,
            prefetch_factor=wandb.config.PREFETCH_FACTOR,
            batch_size=wandb.config.TRAIN_BATCH_SIZE,
            num_workers=wandb.config.WORKERS,
            pin_memory=wandb.config.PIN_MEMORY,
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
