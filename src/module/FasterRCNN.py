import torch
import torchvision
from lightning.pytorch import LightningModule
from PIL import ImageDraw
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights, FastRCNNPredictor


def get_model_instance_segmentation(n_classes: int):
    """Returns a Torchvision FasterRCNN model for finetunning.

    Args:
        n_classes (int): number of classes the model should predict, background excluded
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        box_detections_per_img=10,  # cap numbers of detections, else oom
    )

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes + 1)

    return model


class FasterRCNN(LightningModule):
    """Faster R-CNN Pytorch Lightning Module, encapsulating common PyTorch functions."""

    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        num_labels: int = 3,
    ):
        """Constructor, build model, save hyperparameters."""
        super().__init__()

        # get Mask R-CNN model
        self.net = get_model_instance_segmentation(num_labels)

        # hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_labels = num_labels
        self.save_hyperparameters()

    def forward(self, imgs, **kwargs):
        """Forward pass."""
        return self.net(imgs, **kwargs)

    def common_step(self, batchs, batch_idx):
        # intialize outputs
        outputs = {}

        # for each dataloader
        for dataloader_name, batch in batchs.items():
            # extract pixel_values and labels from batch
            images = batch["pixel_values"]
            targets = batch["labels"]

            # forward pass
            model_output = self(images, targets=targets)

            # get loss
            outputs[dataloader_name] = {
                "loss": sum(model_output.values()),
                "loss_dict": model_output,
            }

        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.common_step(batch, batch_idx)

        # logs metrics for each training_step
        loss = 0
        for dataloader_name, output in outputs.items():
            loss += output["loss"]
            self.log(f"train_loss_{dataloader_name}", output["loss"])
            for k, v in output["loss_dict"].items():
                self.log(f"train_loss_{k}_{dataloader_name}", v.item())

        self.log("lr", self.optimizers().param_groups[0]["lr"])

        return loss

    def validation_step(self, batch, batch_idx):
        self.train()
        with torch.no_grad():
            outputs = self.common_step(batch, batch_idx)
        self.eval()

        # logs metrics for each validation_step
        loss = 0
        for dataloader_name, output in outputs.items():
            loss += output["loss"]
            self.log(f"val_loss_{dataloader_name}", output["loss"])
            for k, v in output["loss_dict"].items():
                self.log(f"val_loss_{k}_{dataloader_name}", v.item())

        return loss

    def configure_optimizers(self):
        """PyTorch optimizers and Schedulers.

        Returns:
            dictionnary for PyTorch Lightning optimizer/scheduler configuration
        """
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        return {
            "optimizer": optimizer,
        }
