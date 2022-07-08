import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import WandbLogger

import wandb
from unet import UNet

CONFIG = {
    "DIR_TRAIN_IMG": "/home/lilian/data_disk/lfainsin/train/",
    "DIR_VALID_IMG": "//home/lilian/data_disk/lfainsin/test_split/",
    "DIR_SPHERE": "/home/lilian/data_disk/lfainsin/spheres+real_split/",
    "FEATURES": [8, 16, 32, 64],
    "N_CHANNELS": 3,
    "N_CLASSES": 1,
    "AMP": True,
    "PIN_MEMORY": True,
    "BENCHMARK": True,
    "DEVICE": "gpu",
    "WORKERS": 10,
    "EPOCHS": 1,
    "BATCH_SIZE": 32,
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

    # create checkpoint callback
    checkpoint_callback = pl.ModelCheckpoint(
        dirpath="checkpoints",
        monitor="val/dice",
    )

    # Create the trainer
    trainer = pl.Trainer(
        max_epochs=CONFIG["EPOCHS"],
        accelerator=CONFIG["DEVICE"],
        # precision=16,
        benchmark=CONFIG["BENCHMARK"],
        val_check_interval=100,
        callbacks=RichProgressBar(),
        logger=logger,
        log_every_n_steps=1,
    )

    trainer.fit(model=net)

    # stop wandb
    wandb.run.finish()
