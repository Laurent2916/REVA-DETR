import logging

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger

from data import Spheres
from mrcnn import MRCNNModule

if __name__ == "__main__":
    # setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # setup wandb
    logger = WandbLogger(
        project="Mask R-CNN",
        config="wandb.yaml",
        save_dir="/tmp/",
        log_model="all",
        settings=wandb.Settings(
            code_dir="./src/",
        ),
    )

    # seed random generators
    pl.seed_everything(
        seed=wandb.config.SEED,
        workers=True,
    )

    # Create Network
    module = MRCNNModule(
        n_classes=2,
    )

    # load checkpoint
    # module.load_state_dict(torch.load()["state_dict"])
    # module.load_from_checkpoint("/tmp/model.ckpt")

    # log gradients and weights regularly
    logger.watch(
        model=module.model,
        log="all",
    )

    # Create the dataloaders
    datamodule = Spheres()

    # Create the trainer
    trainer = pl.Trainer(
        max_epochs=wandb.config.EPOCHS,
        accelerator=wandb.config.DEVICE,
        benchmark=wandb.config.BENCHMARK,
        deterministic=wandb.config.DETERMINISTIC,
        precision=wandb.config.PRECISION,
        logger=logger,
        log_every_n_steps=5,
        val_check_interval=50,
        callbacks=[
            ModelCheckpoint(monitor="valid/loss", mode="min"),
            RichProgressBar(),
        ],
        # profiler="advanced",
        num_sanity_val_steps=3,
        devices=[1],
    )

    # actually train the model
    trainer.fit(model=module, datamodule=datamodule)

    # stop wandb
    wandb.run.finish()  # type: ignore
