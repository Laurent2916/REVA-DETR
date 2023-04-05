from datamodule import DETRDataModule, FasterRCNNDataModule  # noqa: F401
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.cli import LightningCLI
from module import DETR, FasterRCNN  # noqa: F401


class MyLightningCLI(LightningCLI):
    """Custom Lightning CLI to define default arguments."""

    def add_arguments_to_parser(self, parser):
        """Add arguments to parser."""
        parser.set_defaults(
            {
                "trainer.max_steps": 5000,
                "trainer.max_epochs": 1,
                "trainer.accelerator": "gpu",
                "trainer.devices": "[0]",
                "trainer.strategy": "auto",
                "trainer.log_every_n_steps": 25,
                "trainer.val_check_interval": 200,
                "trainer.num_sanity_val_steps": 10,
                "trainer.benchmark": True,
                "trainer.callbacks": [
                    RichProgressBar(),
                    RichModelSummary(max_depth=2),
                    ModelCheckpoint(mode="min", monitor="val_loss_real"),
                    ModelCheckpoint(save_on_train_epoch_end=True),
                ],
            }
        )


if __name__ == "__main__":
    cli = MyLightningCLI(
        model_class=DETR,
        datamodule_class=DETRDataModule,
        seed_everything_default=69420,
    )
