"""Pytorch lightning wrapper for model."""

import pytorch_lightning as pl
import torch
import torchvision
import wandb
from torchvision.models.detection._utils import Matcher
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops.boxes import box_iou


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

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

    def forward(self, imgs):
        # Torchvision FasterRCNN returns the loss during training
        # and the boxes during eval
        self.model.eval()
        return self.model(imgs)

    def training_step(self, batch, batch_idx):
        # unpack batch
        images, targets = batch

        # enable train mode
        self.model.train()

        # fasterrcnn takes both images and targets for training
        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())
        return {"loss": loss, "log": loss_dict}

    def validation_step(self, batch, batch_idx):
        # unpack batch
        images, targets = batch

        # enable eval mode
        self.detector.eval()

        # make a prediction
        preds = self.detector(images)

        # compute validation loss
        self.val_loss = torch.mean(
            torch.stack(
                [
                    self.accuracy(
                        target,
                        pred["boxes"],
                        iou_threshold=0.5,
                    )
                    for target, pred in zip(targets, preds)
                ],
            )
        )

        return self.val_loss

    def accuracy(self, src_boxes, pred_boxes, iou_threshold=1.0):
        """
        The accuracy method is not the one used in the evaluator but very similar
        """
        total_gt = len(src_boxes)
        total_pred = len(pred_boxes)
        if total_gt > 0 and total_pred > 0:

            # Define the matcher and distance matrix based on iou
            matcher = Matcher(iou_threshold, iou_threshold, allow_low_quality_matches=False)
            match_quality_matrix = box_iou(src_boxes, pred_boxes)

            results = matcher(match_quality_matrix)

            true_positive = torch.count_nonzero(results.unique() != -1)
            matched_elements = results[results > -1]

            # in Matcher, a pred element can be matched only twice
            false_positive = torch.count_nonzero(results == -1) + (
                len(matched_elements) - len(matched_elements.unique())
            )
            false_negative = total_gt - true_positive

            return true_positive / (true_positive + false_positive + false_negative)

        elif total_gt == 0:
            if total_pred > 0:
                return torch.tensor(0.0).cuda()
            else:
                return torch.tensor(1.0).cuda()
        elif total_gt > 0 and total_pred == 0:
            return torch.tensor(0.0).cuda()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=wandb.config.LEARNING_RATE,
            momentum=wandb.config.MOMENTUM,
            weight_decay=wandb.config.WEIGHT_DECAY,
            nesterov=wandb.config.NESTEROV,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=3,
            T_mult=1,
            lr=wandb.config.LEARNING_RATE_MIN,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_accuracy",
            },
        }
