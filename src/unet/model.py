""" Full assembly of the parts to form the complete network """

import numpy as np
import pytorch_lightning as pl

import wandb
from utils.dice import dice_coeff

from .blocks import *

class_labels = {
    1: "sphere",
}


class UNet(pl.LightningModule):
    def __init__(self, n_channels, n_classes, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, features[0])

        self.downs = nn.ModuleList()
        for i in range(len(features) - 1):
            self.downs.append(
                Down(*features[i : i + 2]),
            )

        self.ups = nn.ModuleList()
        for i in range(len(features) - 1):
            self.ups.append(
                Up(*features[-1 - i : -3 - i : -1]),
            )

        self.outc = OutConv(features[0], n_classes)

    def forward(self, x):
        skips = []

        x = x.to(self.device)
        x = self.inc(x)

        for down in self.downs:
            skips.append(x)
            x = down(x)

        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        x = self.outc(x)

        return x

    def save_to_table(self, images, masks_true, masks_pred, masks_pred_bin, log_key):
        table = wandb.Table(columns=["ID", "image", "ground truth", "prediction"])

        for i, (img, mask, pred, pred_bin) in enumerate(
            zip(
                images.cpu(),
                masks_true.cpu(),
                masks_pred.cpu(),
                masks_pred_bin.cpu().squeeze(1).int().numpy(),
            )
        ):
            table.add_data(
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
            )

        wandb.log(
            {
                log_key: table,
            }
        )

    def training_step(self, batch, batch_idx):
        # unpacking
        images, masks_true = batch
        masks_true = masks_true.unsqueeze(1)
        masks_pred = self(images)
        masks_pred_bin = (torch.sigmoid(masks_pred) > 0.5).float()

        # compute metrics
        loss = F.cross_entropy(masks_pred, masks_true)
        mae = torch.nn.functional.l1_loss(masks_pred_bin, masks_true)
        accuracy = (masks_true == masks_pred_bin).float().mean()
        dice = dice_coeff(masks_pred_bin, masks_true)

        self.log(
            "train",
            {
                "accuracy": accuracy,
                "bce": loss,
                "dice": dice,
                "mae": mae,
            },
        )

        return loss  # , dice, accuracy, mae

    def validation_step(self, batch, batch_idx):
        # unpacking
        images, masks_true = batch
        masks_true = masks_true.unsqueeze(1)
        masks_pred = self(images)
        masks_pred_bin = (torch.sigmoid(masks_pred) > 0.5).float()

        # compute metrics
        loss = F.cross_entropy(masks_pred, masks_true)
        # mae = torch.nn.functional.l1_loss(masks_pred_bin, masks_true)
        # accuracy = (masks_true == masks_pred_bin).float().mean()
        # dice = dice_coeff(masks_pred_bin, masks_true)

        if batch_idx == 0:
            self.save_to_table(images, masks_true, masks_pred, masks_pred_bin, "val/predictions")

        return loss  # , dice, accuracy, mae

    # def validation_step_end(self, validation_outputs):
    #     # unpacking
    #     loss, dice, accuracy, mae = validation_outputs
    #     # optimizer = self.optimizers[0]
    #     # learning_rate = optimizer.state_dict()["param_groups"][0]["lr"]

    #     wandb.log(
    #         {
    #             # "train/learning_rate": learning_rate,
    #             "val/accuracy": accuracy,
    #             "val/bce": loss,
    #             "val/dice": dice,
    #             "val/mae": mae,
    #         }
    #     )

    #     # export model to onnx
    #     dummy_input = torch.randn(1, 3, 512, 512, requires_grad=True)
    #     torch.onnx.export(self, dummy_input, f"checkpoints/model.onnx")
    #     artifact = wandb.Artifact("onnx", type="model")
    #     artifact.add_file(f"checkpoints/model.onnx")
    #     wandb.run.log_artifact(artifact)

    # def test_step(self, batch, batch_idx):
    #     # unpacking
    #     images, masks_true = batch
    #     masks_true = masks_true.unsqueeze(1)
    #     masks_pred = self(images)
    #     masks_pred_bin = (torch.sigmoid(masks_pred) > 0.5).float()

    #     # compute metrics
    #     loss = F.cross_entropy(masks_pred, masks_true)
    #     mae = torch.nn.functional.l1_loss(masks_pred_bin, masks_true)
    #     accuracy = (masks_true == masks_pred_bin).float().mean()
    #     dice = dice_coeff(masks_pred_bin, masks_true)

    #     if batch_idx == 0:
    #         self.save_to_table(images, masks_true, masks_pred, masks_pred_bin, "test/predictions")

    #     return loss, dice, accuracy, mae

    # def test_step_end(self, test_outputs):
    #     # unpacking
    #     list_loss, list_dice, list_accuracy, list_mae = test_outputs

    #     # averaging
    #     loss = np.mean(list_loss)
    #     dice = np.mean(list_dice)
    #     accuracy = np.mean(list_accuracy)
    #     mae = np.mean(list_mae)

    #     # # get learning rate
    #     # optimizer = self.optimizers[0]
    #     # learning_rate = optimizer.state_dict()["param_groups"][0]["lr"]

    #     wandb.log(
    #         {
    #             # "train/learning_rate": learning_rate,
    #             "test/accuracy": accuracy,
    #             "test/bce": loss,
    #             "test/dice": dice,
    #             "test/mae": mae,
    #         }
    #     )

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(
            self.parameters(),
            lr=wandb.config.LEARNING_RATE,
            weight_decay=wandb.config.WEIGHT_DECAY,
            momentum=wandb.config.MOMENTUM,
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     "max",
        #     patience=2,
        # )

        return optimizer  # , scheduler
