"""Pytorch lightning wrapper for model."""

import itertools

import pytorch_lightning as pl

import wandb
from unet.model import UNet
from utils.dice import dice_loss

from .blocks import *

class_labels = {
    1: "sphere",
}


class UNetModule(pl.LightningModule):
    def __init__(self, n_channels, n_classes, learning_rate, batch_size, features=[64, 128, 256, 512]):
        super(UNetModule, self).__init__()

        # Hyperparameters
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # log hyperparameters
        self.save_hyperparameters()

        # Network
        self.model = UNet(n_channels, n_classes, features)

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch):
        data, ground_truth = batch  # unpacking
        ground_truth = ground_truth.unsqueeze(1)  # 1HW -> HW

        # forward pass, compute masks
        prediction = self.model(data)
        binary = (torch.sigmoid(prediction) > 0.5).float()  # TODO: check if float necessary

        # compute metrics (in dictionnary)
        metrics = {
            "dice": dice_loss(prediction, ground_truth),
            "dice_bin": dice_loss(binary, ground_truth, logits=False),
            "bce": F.binary_cross_entropy_with_logits(prediction, ground_truth),
            "mae": torch.nn.functional.l1_loss(binary, ground_truth),
            "accuracy": (ground_truth == binary).float().mean(),
        }

        # wrap tensors in dictionnary
        tensors = {
            "data": data,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "binary": binary,
        }

        return metrics, tensors

    def training_step(self, batch, batch_idx):
        # compute metrics
        metrics, tensors = self.shared_step(batch)

        # log metrics
        self.log_dict(dict([(f"train/{key}", value) for key, value in metrics.items()]))

        if batch_idx == 5000:
            rows = []
            columns = ["ID", "image", "ground truth", "prediction", "dice", "dice_bin"]
            for i, (img, mask, pred, pred_bin) in enumerate(
                zip(  # TODO: use comprehension list to zip the dictionnary
                    tensors["images"].cpu(),
                    tensors["ground_truth"].cpu(),
                    tensors["prediction"].cpu(),
                    tensors["binary"]
                    .cpu()
                    .squeeze(1)
                    .int()
                    .numpy(),  # TODO: check if .functions can be moved elsewhere
                )
            ):
                rows.append(
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

            # log table
            wandb.log(
                {
                    "train/predictions": wandb.Table(
                        columns=columns,
                        data=rows,
                    )
                }
            )

        return metrics["dice"]

    def validation_step(self, batch, batch_idx):
        metrics, tensors = self.shared_step(batch)

        rows = []
        if batch_idx % 50 == 0 or metrics["dice"] > 0.9:
            for i, (img, mask, pred, pred_bin) in enumerate(
                zip(  # TODO: use comprehension list to zip the dictionnary
                    tensors["images"].cpu(),
                    tensors["ground_truth"].cpu(),
                    tensors["prediction"].cpu(),
                    tensors["binary"]
                    .cpu()
                    .squeeze(1)
                    .int()
                    .numpy(),  # TODO: check if .functions can be moved elsewhere
                )
            ):
                rows.append(
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

        return metrics

    def validation_epoch_end(self, validation_outputs):
        # unpacking
        metricss, rowss = validation_outputs

        # metrics flattening
        metrics = {
            "dice": torch.stack([d["dice"] for d in metricss]).mean(),
            "dice_bin": torch.stack([d["dice_bin"] for d in metricss]).mean(),
            "bce": torch.stack([d["bce"] for d in metricss]).mean(),
            "mae": torch.stack([d["mae"] for d in metricss]).mean(),
            "accuracy": torch.stack([d["accuracy"] for d in validation_outputs]).mean(),
        }

        # log metrics
        self.log_dict(dict([(f"val/{key}", value) for key, value in metrics.items()]))

        # rows flattening
        rows = list(itertools.chain.from_iterable(rowss))
        columns = ["ID", "image", "ground truth", "prediction", "dice", "dice_bin"]

        # log table
        wandb.log(
            {
                "val/predictions": wandb.Table(
                    columns=columns,
                    data=rows,
                )
            }
        )

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=wandb.config.WEIGHT_DECAY,
            momentum=wandb.config.MOMENTUM,
        )

        return optimizer
