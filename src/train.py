import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import WandbLogger

import wandb
from data import Spheres
from unet import UNetModule
from utils import ArtifactLog, TableLog

if __name__ == "__main__":
    # setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # setup wandb
    logger = WandbLogger(
        project="U-Net",
        config="wandb.yaml",
        settings=wandb.Settings(
            code_dir="./src/",
        ),
    )

    # seed random generators
    pl.seed_everything(69420, workers=True)

    # Create network
    model = UNetModule(
        n_channels=wandb.config.N_CHANNELS,
        n_classes=wandb.config.N_CLASSES,
        features=wandb.config.FEATURES,
    )

    # load checkpoint
    # state_dict = torch.load("checkpoints/synth.pth")
    # state_dict = dict([(f"model.{key}", value) for key, value in state_dict.items()])
    # model.load_state_dict(state_dict)

    # log gradients and weights regularly
    logger.watch(model.model, log="all")

    # Create the dataloaders
    datamodule = Spheres()

    # Create the trainer
    trainer = pl.Trainer(
        max_epochs=wandb.config.EPOCHS,
        accelerator=wandb.config.DEVICE,
        benchmark=wandb.config.BENCHMARK,
        precision=16,
        logger=logger,
        log_every_n_steps=1,
        val_check_interval=100,
        callbacks=[RichProgressBar(), ArtifactLog(), TableLog()],
        # profiler="simple",
        # num_sanity_val_steps=0,
    )

    # actually train the model
    trainer.fit(model=model, datamodule=datamodule)

    # stop wandb
    wandb.run.finish()
