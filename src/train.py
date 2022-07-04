import logging

import albumentations as A
import pytorch_lightning as pl
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from src.utils.dataset import SphereDataset
from unet import UNet
from utils.paste import RandomPaste

class_labels = {
    1: "sphere",
}

if __name__ == "__main__":
    # setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # setup wandb
    logger = WandbLogger(
        project="U-Net",
        config=dict(
            DIR_TRAIN_IMG="/home/lilian/data_disk/lfainsin/train/",
            DIR_VALID_IMG="/home/lilian/data_disk/lfainsin/val/",
            DIR_TEST_IMG="/home/lilian/data_disk/lfainsin/test/",
            DIR_SPHERE_IMG="/home/lilian/data_disk/lfainsin/spheres/Images/",
            DIR_SPHERE_MASK="/home/lilian/data_disk/lfainsin/spheres/Masks/",
            FEATURES=[64, 128, 256, 512],
            N_CHANNELS=3,
            N_CLASSES=1,
            AMP=True,
            PIN_MEMORY=True,
            BENCHMARK=True,
            DEVICE="gpu",
            WORKERS=8,
            EPOCHS=5,
            BATCH_SIZE=16,
            LEARNING_RATE=1e-4,
            WEIGHT_DECAY=1e-8,
            MOMENTUM=0.9,
            IMG_SIZE=512,
            SPHERES=5,
        ),
        settings=wandb.Settings(
            code_dir="./src/",
        ),
    )

    # seed random generators
    pl.seed_everything(69420, workers=True)

    # 0. Create network
    net = UNet(n_channels=wandb.config.N_CHANNELS, n_classes=wandb.config.N_CLASSES, features=wandb.config.FEATURES)

    # log the number of parameters of the model
    wandb.config.PARAMETERS = sum(p.numel() for p in net.parameters() if p.requires_grad)

    # log gradients and weights regularly
    logger.watch(net, log="all")

    # 1. Create transforms
    tf_train = A.Compose(
        [
            A.Resize(wandb.config.IMG_SIZE, wandb.config.IMG_SIZE),
            A.Flip(),
            A.ColorJitter(),
            RandomPaste(wandb.config.SPHERES, wandb.config.DIR_SPHERE_IMG, wandb.config.DIR_SPHERE_MASK),
            A.GaussianBlur(),
            A.ISONoise(),
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ],
    )
    tf_valid = A.Compose(
        [
            A.Resize(wandb.config.IMG_SIZE, wandb.config.IMG_SIZE),
            RandomPaste(wandb.config.SPHERES, wandb.config.DIR_SPHERE_IMG, wandb.config.DIR_SPHERE_MASK),
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ],
    )

    # 2. Create datasets
    ds_train = SphereDataset(image_dir=wandb.config.DIR_TRAIN_IMG, transform=tf_train)
    ds_valid = SphereDataset(image_dir=wandb.config.DIR_VALID_IMG, transform=tf_valid)
    ds_test = SphereDataset(image_dir=wandb.config.DIR_TEST_IMG)

    # 2.5. Create subset, if uncommented
    ds_train = torch.utils.data.Subset(ds_train, list(range(0, len(ds_train), len(ds_train) // 10000)))
    ds_valid = torch.utils.data.Subset(ds_valid, list(range(0, len(ds_valid), len(ds_valid) // 1000)))
    ds_test = torch.utils.data.Subset(ds_test, list(range(0, len(ds_test), len(ds_test) // 100)))

    # 3. Create data loaders
    train_loader = DataLoader(
        ds_train,
        shuffle=True,
        batch_size=wandb.config.BATCH_SIZE,
        num_workers=wandb.config.WORKERS,
        pin_memory=wandb.config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        ds_valid,
        shuffle=False,
        drop_last=True,
        batch_size=wandb.config.BATCH_SIZE,
        num_workers=wandb.config.WORKERS,
        pin_memory=wandb.config.PIN_MEMORY,
    )
    test_loader = DataLoader(
        ds_test,
        shuffle=False,
        drop_last=False,
        batch_size=1,
        num_workers=wandb.config.WORKERS,
        pin_memory=wandb.config.PIN_MEMORY,
    )

    # 4. Create the trainer
    trainer = pl.Trainer(
        max_epochs=wandb.config.EPOCHS,
        accelerator="gpu",
        precision=16,
        auto_scale_batch_size="binsearch",
        benchmark=wandb.config.BENCHMARK,
        val_check_interval=100,
    )

    # print the config
    logging.info(f"wandb config:\n{yaml.dump(wandb.config.as_dict())}")

    # # wandb init log
    # wandb.log(
    #     {
    #         "train/learning_rate": optimizer.state_dict()["param_groups"][0]["lr"],
    #     },
    #     commit=False,
    # )

    try:
        trainer.fit(
            model=net,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            test_dataloaders=test_loader,
            accelerator=wandb.config.DEVICE,
        )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), "INTERRUPTED.pth")
        raise

    # stop wandb
    wandb.run.finish()
