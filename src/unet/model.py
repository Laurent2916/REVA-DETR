""" Full assembly of the parts to form the complete network """

import pytorch_lightning as pl

import wandb
from utils.dice import dice_coeff

from .blocks import *

class_labels = {
    1: "sphere",
}


class UNet(pl.LightningModule):
    def __init__(self, n_channels, n_classes, learning_rate, batch_size, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()

        # Hyperparameters
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Network
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

        wandb.log({log_key: table})  # replace by self.log

    def training_step(self, batch, batch_idx):
        # unpacking
        images, masks_true = batch
        masks_true = masks_true.unsqueeze(1)

        # forward pass
        masks_pred = self(images)

        # compute loss
        bce = F.binary_cross_entropy_with_logits(masks_pred, masks_true)

        # compute other metrics
        masks_pred_bin = (torch.sigmoid(masks_pred) > 0.5).float()
        mae = torch.nn.functional.l1_loss(masks_pred_bin, masks_true)
        accuracy = (masks_true == masks_pred_bin).float().mean()
        dice = dice_coeff(masks_pred_bin, masks_true)

        self.log_dict(
            {
                "train/accuracy": accuracy,
                "train/bce": bce,
                "train/dice": dice,
                "train/mae": mae,
            },
        )

        return dict(
            loss=bce,
            dice=dice,
            accuracy=accuracy,
            mae=mae,
        )

    def validation_step(self, batch, batch_idx):
        # unpacking
        images, masks_true = batch
        masks_true = masks_true.unsqueeze(1)

        # forward pass
        masks_pred = self(images)

        # compute loss
        bce = F.binary_cross_entropy_with_logits(masks_pred, masks_true)

        # compute other metrics
        masks_pred_bin = (torch.sigmoid(masks_pred) > 0.5).float()
        mae = torch.nn.functional.l1_loss(masks_pred_bin, masks_true)
        accuracy = (masks_true == masks_pred_bin).float().mean()
        dice = dice_coeff(masks_pred_bin, masks_true)

        if batch_idx == 0:
            self.save_to_table(images, masks_true, masks_pred, masks_pred_bin, "val/predictions")

        return dict(
            loss=bce,
            dice=dice,
            accuracy=accuracy,
            mae=mae,
        )

    def validation_epoch_end(self, validation_outputs):
        # unpacking
        accuracy = torch.stack([d["accuracy"] for d in validation_outputs]).mean()
        loss = torch.stack([d["loss"] for d in validation_outputs]).mean()
        dice = torch.stack([d["dice"] for d in validation_outputs]).mean()
        mae = torch.stack([d["mae"] for d in validation_outputs]).mean()

        # logging
        wandb.log(
            {
                "val/accuracy": accuracy,
                "val/bce": loss,
                "val/dice": dice,
                "val/mae": mae,
            }
        )

        # export model to pth
        torch.save(self.state_dict(), f"checkpoints/model.pth")
        artifact = wandb.Artifact("pth", type="model")
        artifact.add_file(f"checkpoints/model.pth")
        wandb.run.log_artifact(artifact)

        # export model to onnx
        dummy_input = torch.randn(1, 3, 512, 512, requires_grad=True)
        torch.onnx.export(self, dummy_input, f"checkpoints/model.onnx")
        artifact = wandb.Artifact("onnx", type="model")
        artifact.add_file(f"checkpoints/model.onnx")
        wandb.run.log_artifact(artifact)

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

        return optimizer
