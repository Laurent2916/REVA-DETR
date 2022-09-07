import wandb
from pytorch_lightning.callbacks import Callback

columns = [
    "ID",
    "image",
]
class_labels = {
    1: "sphere",
}


class TableLog(Callback):
    def on_validation_epoch_start(self, trainer, pl_module):
        self.rows = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # unpacking
        if batch_idx == 2:
            images, targets = batch

            for i, (image, target, pred) in enumerate(
                zip(
                    images,
                    targets,
                    outputs,
                )
            ):
                self.rows.append(
                    [
                        i,
                        wandb.Image(
                            image.cpu(),
                            masks={
                                "ground_truth": {
                                    "mask_data": (target["masks"].cpu().sum(dim=0) > 0.5).int().numpy(),
                                    "class_labels": class_labels,
                                },
                                "predictions": {
                                    "mask_data": (pred["masks"].cpu().sum(dim=0) > 0.5).int().numpy(),
                                    "class_labels": class_labels,
                                },
                            },
                        ),
                    ]
                )

    def on_validation_epoch_end(self, trainer, pl_module):
        # log table
        wandb.log(
            {
                "valid/predictions": wandb.Table(
                    columns=columns,
                    data=self.rows,
                )
            }
        )
