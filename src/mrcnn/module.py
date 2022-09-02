"""Pytorch lightning wrapper for model."""

import pytorch_lightning as pl
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import (
    MaskRCNN_ResNet50_FPN_Weights,
    MaskRCNNPredictor,
)

import wandb
from utils.coco_eval import CocoEvaluator
from utils.coco_utils import get_coco_api_from_dataset, get_iou_types


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

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
    def __init__(self, hidden_layer_size, n_classes):
        super().__init__()

        # Hyperparameters
        self.hidden_layers_size = hidden_layer_size
        self.n_classes = n_classes

        # log hyperparameters
        self.save_hyperparameters()

        # Network
        self.model = get_model_instance_segmentation(n_classes)

        # pycoco evaluator
        self.coco = None
        self.iou_types = get_iou_types(self.model)
        self.coco_evaluator = None

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
        if self.coco is None:
            self.coco = get_coco_api_from_dataset(self.trainer.val_dataloaders[0].dataset)

        # init coco evaluator
        self.coco_evaluator = CocoEvaluator(self.coco, self.iou_types)

    def validation_step(self, batch, batch_idx):
        # unpack batch
        images, targets = batch

        # compute metrics using pycocotools
        outputs = self.model(images)
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        self.coco_evaluator.update(res)

        # compute validation loss
        self.model.train()
        loss_dict = self.model(images, targets)
        loss_dict = {f"valid/{key}": val for key, val in loss_dict.items()}
        loss_dict["valid/loss"] = sum(loss_dict.values())
        self.model.eval()

        return loss_dict

    def validation_epoch_end(self, outputs):
        # log validation loss
        loss_dict = {k: torch.stack([d[k] for d in outputs]).mean() for k in outputs[0].keys()}
        self.log_dict(loss_dict)

        # accumulate all predictions
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()

        YEET = {
            "valid,bbox,AP,IoU=0.50:0.,area=all,maxDets=100": self.coco_evaluator.coco_eval["bbox"].stats[0],
            "valid,bbox,AP,IoU=0.50,area=all,maxDets=100": self.coco_evaluator.coco_eval["bbox"].stats[1],
            "valid,bbox,AP,IoU=0.75,area=all,maxDets=100": self.coco_evaluator.coco_eval["bbox"].stats[2],
            "valid,bbox,AP,IoU=0.50:0.,area=small,maxDets=100": self.coco_evaluator.coco_eval["bbox"].stats[3],
            "valid,bbox,AP,IoU=0.50:0.,area=medium,maxDets=100": self.coco_evaluator.coco_eval["bbox"].stats[4],
            "valid,bbox,AP,IoU=0.50:0.,area=large,maxDets=100": self.coco_evaluator.coco_eval["bbox"].stats[5],
            "valid,bbox,AR,IoU=0.50:0.,area=all,maxDets=1": self.coco_evaluator.coco_eval["bbox"].stats[6],
            "valid,bbox,AR,IoU=0.50:0.,area=all,maxDets=10": self.coco_evaluator.coco_eval["bbox"].stats[7],
            "valid,bbox,AR,IoU=0.50:0.,area=all,maxDets=100": self.coco_evaluator.coco_eval["bbox"].stats[8],
            "valid,bbox,AR,IoU=0.50:0.,area=small,maxDets=100": self.coco_evaluator.coco_eval["bbox"].stats[9],
            "valid,bbox,AR,IoU=0.50:0.,area=medium,maxDets=100": self.coco_evaluator.coco_eval["bbox"].stats[10],
            "valid,bbox,AR,IoU=0.50:0.,area=large,maxDets=100": self.coco_evaluator.coco_eval["bbox"].stats[11],
            "valid,segm,AP,IoU=0.50:0.,area=all,maxDets=100": self.coco_evaluator.coco_eval["segm"].stats[0],
            "valid,segm,AP,IoU=0.50,area=all,maxDets=100": self.coco_evaluator.coco_eval["segm"].stats[1],
            "valid,segm,AP,IoU=0.75,area=all,maxDets=100": self.coco_evaluator.coco_eval["segm"].stats[2],
            "valid,segm,AP,IoU=0.50:0.,area=small,maxDets=100": self.coco_evaluator.coco_eval["segm"].stats[3],
            "valid,segm,AP,IoU=0.50:0.,area=medium,maxDets=100": self.coco_evaluator.coco_eval["segm"].stats[4],
            "valid,segm,AP,IoU=0.50:0.,area=large,maxDets=100": self.coco_evaluator.coco_eval["segm"].stats[5],
            "valid,segm,AR,IoU=0.50:0.,area=all,maxDets=1": self.coco_evaluator.coco_eval["segm"].stats[6],
            "valid,segm,AR,IoU=0.50:0.,area=all,maxDets=10": self.coco_evaluator.coco_eval["segm"].stats[7],
            "valid,segm,AR,IoU=0.50:0.,area=all,maxDets=100": self.coco_evaluator.coco_eval["segm"].stats[8],
            "valid,segm,AR,IoU=0.50:0.,area=small,maxDets=100": self.coco_evaluator.coco_eval["segm"].stats[9],
            "valid,segm,AR,IoU=0.50:0.,area=medium,maxDets=100": self.coco_evaluator.coco_eval["segm"].stats[10],
            "valid,segm,AR,IoU=0.50:0.,area=large,maxDets=100": self.coco_evaluator.coco_eval["segm"].stats[11],
        }

        self.log_dict(YEET)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=wandb.config.LEARNING_RATE,
            momentum=wandb.config.MOMENTUM,
            weight_decay=wandb.config.WEIGHT_DECAY,
        )

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=3,
        #     T_mult=1,
        #     lr=wandb.config.LEARNING_RATE_MIN,
        #     verbose=True,
        # )

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "val_accuracy",
        #     },
        # }

        return optimizer
