import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger

import wandb
from data import Spheres
from unet import UNetModule

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
    "WORKERS": 14,
    "EPOCHS": 1,
    "BATCH_SIZE": 16 * 3,
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
    model = UNetModule(
        n_channels=CONFIG["N_CHANNELS"],
        n_classes=CONFIG["N_CLASSES"],
        batch_size=CONFIG["BATCH_SIZE"],
        learning_rate=CONFIG["LEARNING_RATE"],
        features=CONFIG["FEATURES"],
    )

    # log gradients and weights regularly
    logger.watch(model, log="all")

    # create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        monitor="val/dice",
    )

    # Create the dataloaders
    datamodule = Spheres()

    # Create the trainer
    trainer = pl.Trainer(
        max_epochs=CONFIG["EPOCHS"],
        accelerator=CONFIG["DEVICE"],
        benchmark=CONFIG["BENCHMARK"],
        # profiler="simple",
        # precision=16,
        logger=logger,
        log_every_n_steps=1,
        val_check_interval=100,
        callbacks=RichProgressBar(),
    )

    trainer.fit(model=model, datamodule=datamodule)

    # stop wandb
    wandb.run.finish()
