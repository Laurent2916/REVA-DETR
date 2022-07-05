import logging

import albumentations as A
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from src.utils.dataset import SphereDataset
from unet import UNet
from utils.paste import RandomPaste

CONFIG = {
    "DIR_TRAIN_IMG": "/home/lilian/data_disk/lfainsin/train/",
    "DIR_VALID_IMG": "/home/lilian/data_disk/lfainsin/val/",
    "DIR_TEST_IMG": "/home/lilian/data_disk/lfainsin/test/",
    "DIR_SPHERE_IMG": "/home/lilian/data_disk/lfainsin/spheres/Images/",
    "DIR_SPHERE_MASK": "/home/lilian/data_disk/lfainsin/spheres/Masks/",
    "FEATURES": [64, 128, 256, 512],
    "N_CHANNELS": 3,
    "N_CLASSES": 1,
    "AMP": True,
    "PIN_MEMORY": True,
    "BENCHMARK": True,
    "DEVICE": "gpu",
    "WORKERS": 8,
    "EPOCHS": 5,
    "BATCH_SIZE": 16,
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 1e-8,
    "MOMENTUM": 0.9,
    "IMG_SIZE": 512,
    "SPHERES": 5,
}

if __name__ == "__main__":
    # setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # setup wandb
    logger = WandbLogger(
        project="U-Net",
        config=CONFIG,
        settings=wandb.Settings(
            code_dir="./src/",
        ),
    )

    # seed random generators
    pl.seed_everything(69420, workers=True)

    # 0. Create network
    net = UNet(n_channels=CONFIG["N_CHANNELS"], n_classes=CONFIG["N_CLASSES"], features=CONFIG["FEATURES"])

    # log gradients and weights regularly
    logger.watch(net, log="all")

    # 1. Create transforms
    tf_train = A.Compose(
        [
            A.Resize(CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]),
            A.Flip(),
            A.ColorJitter(),
            RandomPaste(CONFIG["SPHERES"], CONFIG["DIR_SPHERE_IMG"], CONFIG["DIR_SPHERE_MASK"]),
            A.GaussianBlur(),
            A.ISONoise(),
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ],
    )

    # 2. Create datasets
    ds_train = SphereDataset(image_dir=CONFIG["DIR_TRAIN_IMG"], transform=tf_train)
    ds_valid = SphereDataset(image_dir=CONFIG["DIR_TEST_IMG"])

    # 2.5. Create subset, if uncommented
    ds_train = torch.utils.data.Subset(ds_train, list(range(0, len(ds_train), len(ds_train) // 10000)))
    # ds_valid = torch.utils.data.Subset(ds_valid, list(range(0, len(ds_valid), len(ds_valid) // 100)))
    # ds_test = torch.utils.data.Subset(ds_test, list(range(0, len(ds_test), len(ds_test) // 100)))

    # 3. Create data loaders
    train_loader = DataLoader(
        ds_train,
        shuffle=True,
        batch_size=CONFIG["BATCH_SIZE"],
        num_workers=CONFIG["WORKERS"],
        pin_memory=CONFIG["PIN_MEMORY"],
    )
    val_loader = DataLoader(
        ds_valid,
        shuffle=False,
        drop_last=True,
        batch_size=1,
        num_workers=CONFIG["WORKERS"],
        pin_memory=CONFIG["PIN_MEMORY"],
    )

    # 4. Create the trainer
    trainer = pl.Trainer(
        max_epochs=CONFIG["EPOCHS"],
        accelerator=CONFIG["DEVICE"],
        # precision=16,
        auto_scale_batch_size="binsearch",
        benchmark=CONFIG["BENCHMARK"],
        val_check_interval=100,
        callbacks=RichProgressBar(),
    )

    try:
        trainer.fit(
            model=net,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), "INTERRUPTED.pth")
        raise

    # stop wandb
    wandb.run.finish()
