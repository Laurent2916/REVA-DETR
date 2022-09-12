"""Mask R-CNN Pytorch Lightning Module for Object Detection and Segmentation."""

from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
import torchvision
import wandb
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import (
    MaskRCNN,
    MaskRCNN_ResNet50_FPN_Weights,
    MaskRCNNPredictor,
)

Prediction = List[Dict[str, torch.Tensor]]


def get_model_instance_segmentation(n_classes: int) -> MaskRCNN:
    """Returns a Torchvision MaskRCNN model for finetunning.

    Args:
        n_classes (int): number of classes the model should predict, background included

    Returns:
        MaskRCNN: the model ready to be used
    """
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
        box_detections_per_img=10,  # cap numbers of detections, else memory explosion
    )

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, n_classes)

    return model


class MRCNNModule(pl.LightningModule):
    """Mask R-CNN Pytorch Lightning Module encapsulating commong PyTorch functions."""

    def __init__(self, n_classes: int) -> None:
        """Constructor, build model, save hyperparameters.

        Args:
            n_classes (int): number of classes the model should predict, background included
        """
        super().__init__()

        # Hyperparameters
        self.n_classes = n_classes

        # log hyperparameters
        self.save_hyperparameters()

        # Network
        self.model = get_model_instance_segmentation(n_classes)

        # onnx export
        self.example_input_array = torch.randn(1, 3, 1024, 1024, requires_grad=True).half()

        # torchmetrics
        self.metric_bbox = MeanAveragePrecision(iou_type="bbox")
        self.metric_segm = MeanAveragePrecision(iou_type="segm")

    def forward(self, imgs: torch.Tensor) -> Prediction:  # type: ignore
        """Make a forward pass (prediction), usefull for onnx export.

        Args:
            imgs (torch.Tensor): the images whose prediction we wish to make

        Returns:
            torch.Tensor: the predictions
        """
        self.model.eval()
        pred: Prediction = self.model(imgs)
        return pred

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:  # type: ignore
        """PyTorch training step.

        Args:
            batch (torch.Tensor): the batch to train the model on
            batch_idx (int): the batch index number

        Returns:
            float: the training loss of this step
        """
        # unpack batch
        images, targets = batch

        # compute loss
        loss_dict: dict[str, float] = self.model(images, targets)
        loss_dict = {f"train/{key}": val for key, val in loss_dict.items()}
        loss = sum(loss_dict.values())
        loss_dict["train/loss"] = loss

        # log everything
        self.log_dict(loss_dict)

        return loss

    def on_validation_epoch_start(self) -> None:
        """Reset TorchMetrics."""
        self.metric_bbox.reset()
        self.metric_segm.reset()

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Prediction:  # type: ignore
        """PyTorch validation step.

        Args:
            batch (torch.Tensor): the batch to evaluate the model on
            batch_idx (int): the batch index number

        Returns:
            torch.Tensor: the predictions
        """
        # unpack batch
        images, targets = batch

        # make prediction
        preds: Prediction = self.model(images)

        # update TorchMetrics from predictions
        for pred, target in zip(preds, targets):
            pred["masks"] = pred["masks"].squeeze(1).int().bool()
            target["masks"] = target["masks"].squeeze(1).int().bool()
        self.metric_bbox.update(preds, targets)
        self.metric_segm.update(preds, targets)

        return preds

    def validation_epoch_end(self, outputs: List[Prediction]) -> None:  # type: ignore
        """Compute TorchMetrics.

        Args:
            outputs (List[Prediction]): list of predictions from validation steps
        """
        # compute and log bounding boxes metrics
        metric_dict = self.metric_bbox.compute()
        metric_dict = {f"valid/bbox/{key}": val for key, val in metric_dict.items()}
        self.log_dict(metric_dict)

        # compute and log semgentation metrics
        metric_dict = self.metric_segm.compute()
        metric_dict = {f"valid/segm/{key}": val for key, val in metric_dict.items()}
        self.log_dict(metric_dict)

    def configure_optimizers(self) -> Dict[str, Any]:
        """PyTorch optimizers and Schedulers.

        Returns:
            Dict[str, Any]: dictionnary for PyTorch Lightning optimizer/scheduler configuration
        """
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
