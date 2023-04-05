import torch
from lightning.pytorch import LightningModule
from PIL import ImageDraw
from transformers import (
    DetrForObjectDetection,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)


class DETR(LightningModule):
    """PyTorch Lightning module for DETR."""

    def __init__(
        self,
        lr: float = 1e-4,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
        num_queries: int = 100,
        warmup_steps: int = 0,
        num_labels: int = 3,
        prediction_threshold: float = 0.9,
    ):
        """Constructor.

        Args:
            lr (float, optional): Learning rate.
            lr_backbone (float, optional): Learning rate for backbone.
            weight_decay (float, optional): Weight decay.
            num_queries (int, optional): Number of queries.
            warmup_steps (int, optional): Number of warmup steps.
            num_labels (int, optional): Number of labels.
            prediction_threshold (float, optional): Prediction threshold.
        """
        super().__init__()

        # get DETR model
        self.net = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            ignore_mismatched_sizes=True,
            num_queries=num_queries,
            num_labels=num_labels,
        )
        torch.compile(self.net)

        # cf https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.prediction_threshold = prediction_threshold
        self.save_hyperparameters()

    def forward(self, pixel_values, pixel_mask, **kwargs):
        """Forward pass."""
        return self.net(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            **kwargs,
        )

    def common_step(self, batchs, batch_idx):
        """Common step for training and validation.

        Args:
            batch (dict): Batch from dataloader (after collate_fn).
            Structure is similar to the following:
            {
                "pixel_values": TensorType["batch", "canal", "width", "height"],
                "pixel_mask": TensorType["batch", 1200, 1200],
                "labels": List[Dict[str, TensorType["batch", "num_boxes", "num_labels"]]], # TODO: check this type
            }

            batch_idx (int): Batch index.

        Returns:
            tuple: Loss and loss dict.
        """
        # intialize outputs
        outputs = {k: {"loss": None, "loss_dict": None} for k in batchs.keys()}

        # for each dataloader
        for dataloader_name, batch in batchs.items():
            # extract pixel_values, pixel_mask and labels from batch
            pixel_values = batch["pixel_values"]
            pixel_mask = batch["pixel_mask"]
            labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

            # forward pass
            model_output = self(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

            # get loss
            outputs[dataloader_name] = {
                "loss": model_output.loss,
                "loss_dict": model_output.loss_dict,
            }

        return outputs

    def training_step(self, batch, batch_idx):
        """Training step."""
        outputs = self.common_step(batch, batch_idx)

        # logs metrics for each training_step
        loss = 0
        for dataloader_name, output in outputs.items():
            loss += output["loss"]
            self.log(f"train_loss_{dataloader_name}", output["loss"])
            for k, v in output["loss_dict"].items():
                self.log(f"train_loss_{k}_{dataloader_name}", v.item())

        self.log("lr", self.optimizers().param_groups[0]["lr"])
        self.log("lr_backbone", self.optimizers().param_groups[1]["lr"])

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        """Validation step."""
        outputs = self.common_step(batch, batch_idx)

        # logs metrics for each validation_step
        loss = 0
        for dataloader_name, output in outputs.items():
            loss += output["loss"]
            self.log(f"val_loss_{dataloader_name}", output["loss"])
            for k, v in output["loss_dict"].items():
                self.log(f"val_loss_{k}_{dataloader_name}", v.item())

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """Predict step."""
        # extract pixel_values and pixelmask from batch
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        images = batch["images"]

        from transformers import AutoImageProcessor

        image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")

        # forward pass
        outputs = self(pixel_values=pixel_values, pixel_mask=pixel_mask)

        # postprocess outputs
        sizes = torch.tensor([image.size[::-1] for image in images], device=self.device)
        processed_outputs = image_processor.post_process_object_detection(
            outputs, threshold=self.prediction_threshold, target_sizes=sizes
        )

        for i, image in enumerate(images):
            # create ImageDraw object to draw on image
            draw = ImageDraw.Draw(image)

            # draw predicted bboxes
            for bbox, label, score in zip(
                processed_outputs[i]["boxes"].cpu().detach().numpy(),
                processed_outputs[i]["labels"].cpu().detach().numpy(),
                processed_outputs[i]["scores"].cpu().detach().numpy(),
            ):
                if label == 0:
                    outline = "red"
                elif label == 1:
                    outline = "blue"
                else:
                    outline = "green"
                draw.rectangle(bbox, outline=outline, width=5)
                draw.text((bbox[0], bbox[1]), f"{score:0.4f}", fill="black", width=15)

            # save image to image.png using PIL
            image.save(f"image2_{batch_idx}_{i}.jpg")

    def configure_optimizers(self):
        """Configure optimizers."""
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad],
            },
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
