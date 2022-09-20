"""Main script, to be launched to start the fine tuning of the neural network."""

import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

from data import Spheres
from modules import MRCNNModule
from utils.callback import TableLog

if __name__ == "__main__":
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
        val_check_interval=250,
        callbacks=[
            EarlyStopping(monitor="valid/sum/map", mode="max", patience=10, min_delta=0.01),
            ModelCheckpoint(monitor="valid/sum/map", mode="max"),
            # ModelPruning("l1_unstructured", amount=0.5),
            LearningRateMonitor(log_momentum=True),
            # StochasticWeightAveraging(swa_lrs=1e-2),
            RichModelSummary(max_depth=2),
            RichProgressBar(),
            TableLog(),
        ],
        # profiler="advanced",
        gradient_clip_val=1,
        num_sanity_val_steps=3,
        devices=[0],
    )

    # actually train the model
    trainer.fit(model=module, datamodule=datamodule)

    # stop wandb
    wandb.run.finish()  # type: ignore
