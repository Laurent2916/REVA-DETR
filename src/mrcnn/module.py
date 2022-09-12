"""Pytorch lightning wrapper for model."""

import pytorch_lightning as pl
import torch
import torchvision
import wandb
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import (
    MaskRCNN_ResNet50_FPN_Weights,
    MaskRCNNPredictor,
)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
        box_detections_per_img=10,  # cap numbers of detections, else memory explosion
    )

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


class MRCNNModule(pl.LightningModule):
    def __init__(self, n_classes):
        super().__init__()

        # Hyperparameters
        self.n_classes = n_classes

        # log hyperparameters
        self.save_hyperparameters()

        # Network
        self.model = get_model_instance_segmentation(n_classes)

        # onnx export
        self.example_input_array = torch.randn(1, 3, 1024, 1024, requires_grad=True).half()

    def forward(self, imgs):
        self.model.eval()
        return self.model(imgs)

    def training_step(self, batch, batch_idx):
        # unpack batch
        images, targets = batch

        # compute loss
        loss_dict = self.model(images, targets)
        loss_dict = {f"train/{key}": val for key, val in loss_dict.items()}
        loss = sum(loss_dict.values())
        loss_dict["train/loss"] = loss

        # log everything
        self.log_dict(loss_dict)

        return loss

    def on_validation_epoch_start(self):
        self.metric_bbox = MeanAveragePrecision(iou_type="bbox")
        self.metric_segm = MeanAveragePrecision(iou_type="segm")

    def validation_step(self, batch, batch_idx):
        # unpack batch
        images, targets = batch

        preds = self.model(images)
        for pred, target in zip(preds, targets):
            pred["masks"] = pred["masks"].squeeze(1).int().bool()
            target["masks"] = target["masks"].squeeze(1).int().bool()
        self.metric_bbox.update(preds, targets)
        self.metric_segm.update(preds, targets)

        return preds

    def validation_epoch_end(self, outputs):
        # log metrics
        metric_dict = self.metric_bbox.compute()
        metric_dict = {f"valid/bbox/{key}": val for key, val in metric_dict.items()}
        self.log_dict(metric_dict)

        metric_dict = self.metric_segm.compute()
        metric_dict = {f"valid/segm/{key}": val for key, val in metric_dict.items()}
        self.log_dict(metric_dict)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=wandb.config.LEARNING_RATE,
            # momentum=wandb.config.MOMENTUM,
            # weight_decay=wandb.config.WEIGHT_DECAY,
        )

        # scheduler = LinearWarmupCosineAnnealingLR(
        #     optimizer,
        #     warmup_epochs=1,
        #     max_epochs=30,
        # )

        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "interval": "step",
            #     "frequency": 10,
            #     "monitor": "bbox/map",
            # },
        }
