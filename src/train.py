import logging

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from unet import UNet

CONFIG = {
    "DIR_TRAIN_IMG": "/home/lilian/data_disk/lfainsin/train/",
    "DIR_VALID_IMG": "/home/lilian/data_disk/lfainsin/val/",
    "DIR_TEST_IMG": "/home/lilian/data_disk/lfainsin/test/",
    "DIR_SPHERE_IMG": "/home/lilian/data_disk/lfainsin/spheres/Images/",
    "DIR_SPHERE_MASK": "/home/lilian/data_disk/lfainsin/spheres/Masks/",
    "FEATURES": [16, 32, 64, 128],
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

    # Create network
    net = UNet(
        n_channels=CONFIG["N_CHANNELS"],
        n_classes=CONFIG["N_CLASSES"],
        batch_size=CONFIG["BATCH_SIZE"],
        learning_rate=CONFIG["LEARNING_RATE"],
        features=CONFIG["FEATURES"],
    )

    # log gradients and weights regularly
    logger.watch(net, log="all")

    # Create the trainer
    trainer = pl.Trainer(
        max_epochs=CONFIG["EPOCHS"],
        accelerator=CONFIG["DEVICE"],
        # precision=16,
        # auto_scale_batch_size="binsearch",
        # auto_lr_find=True,
        benchmark=CONFIG["BENCHMARK"],
        val_check_interval=100,
        callbacks=RichProgressBar(),
        logger=logger,
        log_every_n_steps=1,
    )

    try:
        trainer.tune(net)
        trainer.fit(model=net)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), "INTERRUPTED.pth")
        raise

    # stop wandb
    wandb.run.finish()
