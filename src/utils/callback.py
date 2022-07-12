import numpy as np
import torch
from pytorch_lightning.callbacks import Callback

import wandb

columns = [
    "ID",
    "image",
    "ground truth",
    "prediction",
    "dice",
    "dice_bin",
]
class_labels = {
    1: "sphere",
}


class TableLog(Callback):
    def on_validation_epoch_start(self, trainer, pl_module):
        self.rows = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # unpacking
        if batch_idx == 0:
            images, ground_truth = batch
            metrics, predictions = outputs

            for i, (img, mask, pred, pred_bin) in enumerate(
                zip(
                    images.cpu(),
                    ground_truth.cpu(),
                    predictions["linear"].cpu().float(),
                    predictions["binary"].cpu().squeeze(1).int().numpy(),
                )
            ):
                self.rows.append(
                    [
                        i,
                        wandb.Image(img),
                        wandb.Image(mask),
                        wandb.Image(
                            pred,
                            masks={
                                "predictions": {
                                    "mask_data": pred_bin,
                                    "class_labels": class_labels,
                                },
                            },
                        ),
                        metrics["dice"],
                        metrics["dice_bin"],
                    ]
                )

    def on_validation_epoch_end(self, trainer, pl_module):
        # log table
        wandb.log(
            {
                "val/predictions": wandb.Table(
                    columns=columns,
                    data=self.rows,
                )
            }
        )


class ArtifactLog(Callback):
    def on_fit_start(self, trainer, pl_module):
        self.best = 1

    def on_validation_epoch_start(self, trainer, pl_module):
        self.dices = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # unpacking
        metrics, _ = outputs
        self.dices.append(metrics["dice"].cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        dice = np.mean(self.dices)

        if dice < self.best:
            self.best = dice

            # create checkpoint
            trainer.save_checkpoint("checkpoints/model.ckpt")

            # log artifact
            artifact = wandb.Artifact("ckpt", type="model")
            artifact.add_file("checkpoints/model.ckpt")
            wandb.run.log_artifact(artifact)
