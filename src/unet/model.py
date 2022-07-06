""" Full assembly of the parts to form the complete network """

import itertools

import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

import wandb
from src.utils.dataset import SphereDataset
from utils.dice import dice_loss
from utils.paste import RandomPaste

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

        # log hyperparameters
        self.save_hyperparameters()

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

    def train_dataloader(self):
        tf_train = A.Compose(
            [
                A.Resize(wandb.config.IMG_SIZE, wandb.config.IMG_SIZE),
                A.Flip(),
                A.ColorJitter(),
                RandomPaste(wandb.config.SPHERES, wandb.config.DIR_SPHERE_IMG, wandb.config.DIR_SPHERE_MASK),
                A.GaussianBlur(),
                A.ISONoise(),
                A.ToFloat(max_value=255),
                ToTensorV2(),
            ],
        )

        ds_train = SphereDataset(image_dir=wandb.config.DIR_TRAIN_IMG, transform=tf_train)
        ds_train = torch.utils.data.Subset(ds_train, list(range(0, len(ds_train), len(ds_train) // 5000)))

        return DataLoader(
            ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=wandb.config.WORKERS,
            pin_memory=wandb.config.PIN_MEMORY,
        )

    def val_dataloader(self):
        ds_valid = SphereDataset(image_dir=wandb.config.DIR_TEST_IMG)

        return DataLoader(
            ds_valid,
            shuffle=False,
            batch_size=1,
            num_workers=wandb.config.WORKERS,
            pin_memory=wandb.config.PIN_MEMORY,
        )

    def training_step(self, batch, batch_idx):
        # unpacking
        images, masks_true = batch
        masks_true = masks_true.unsqueeze(1)

        # forward pass
        masks_pred = self(images)

        # compute metrics
        bce = F.binary_cross_entropy_with_logits(masks_pred, masks_true)
        dice = dice_loss(masks_pred, masks_true)

        masks_pred_bin = (torch.sigmoid(masks_pred) > 0.5).float()
        dice_bin = dice_loss(masks_pred_bin, masks_true, logits=False)
        mae = torch.nn.functional.l1_loss(masks_pred_bin, masks_true)
        accuracy = (masks_true == masks_pred_bin).float().mean()

        self.log_dict(
            {
                "train/accuracy": accuracy,
                "train/dice": dice,
                "train/dice_bin": dice_bin,
                "train/bce": bce,
                "train/mae": mae,
            },
        )

        return dict(
            accuracy=accuracy,
            loss=dice,
            bce=bce,
            mae=mae,
        )

    def validation_step(self, batch, batch_idx):
        # unpacking
        images, masks_true = batch
        masks_true = masks_true.unsqueeze(1)

        # forward pass
        masks_pred = self(images)

        # compute metrics
        bce = F.binary_cross_entropy_with_logits(masks_pred, masks_true)
        dice = dice_loss(masks_pred, masks_true)

        masks_pred_bin = (torch.sigmoid(masks_pred) > 0.5).float()
        dice_bin = dice_loss(masks_pred_bin, masks_true, logits=False)
        mae = torch.nn.functional.l1_loss(masks_pred_bin, masks_true)
        accuracy = (masks_true == masks_pred_bin).float().mean()

        rows = []
        if batch_idx % 50 == 0:
            for i, (img, mask, pred, pred_bin) in enumerate(
                zip(
                    images.cpu(),
                    masks_true.cpu(),
                    masks_pred.cpu(),
                    masks_pred_bin.cpu().squeeze(1).int().numpy(),
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
                    ]
                )

        return dict(
            accuracy=accuracy,
            loss=dice,
            dice_bin=dice_bin,
            bce=bce,
            mae=mae,
            table_rows=rows,
        )

    def validation_epoch_end(self, validation_outputs):
        # matrics unpacking
        accuracy = torch.stack([d["accuracy"] for d in validation_outputs]).mean()
        dice_bin = torch.stack([d["dice_bin"] for d in validation_outputs]).mean()
        loss = torch.stack([d["loss"] for d in validation_outputs]).mean()
        bce = torch.stack([d["bce"] for d in validation_outputs]).mean()
        mae = torch.stack([d["mae"] for d in validation_outputs]).mean()

        # table unpacking
        columns = ["ID", "image", "ground truth", "prediction"]
        rowss = [d["table_rows"] for d in validation_outputs]
        rows = list(itertools.chain.from_iterable(rowss))

        # logging
        try:  # required by autofinding, logger replaced by dummy
            self.logger.log_table(
                key="val/predictions",
                columns=columns,
                data=rows,
            )
        except:
            pass

        self.log_dict(
            {
                "val/accuracy": accuracy,
                "val/dice": loss,
                "val/dice_bin": dice_bin,
                "val/bce": bce,
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

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=wandb.config.WEIGHT_DECAY,
            momentum=wandb.config.MOMENTUM,
        )

        return optimizer
