import wandb
from pytorch_lightning.callbacks import Callback

columns = [
    "image",
]
class_labels = {
    1: "sphere",
    2: "chrome",
    10: "sphere_gt",
    20: "chrome_gt",
}


class TableLog(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            rows = []

            # unpacking
            images, targets = batch

            for image, target in zip(
                images,
                targets,
            ):
                rows.append(
                    [
                        wandb.Image(
                            image.cpu(),
                            masks={
                                "ground_truth": {
                                    "mask_data": (target["masks"] * target["labels"][:, None, None])
                                    .max(dim=0)
                                    .values.mul(10)
                                    .cpu()
                                    .numpy(),
                                    "class_labels": class_labels,
                                },
                            },
                        ),
                    ]
                )

            wandb.log(
                {
                    "train/predictions": wandb.Table(
                        columns=columns,
                        data=rows,
                    )
                }
            )

    def on_validation_epoch_start(self, trainer, pl_module):
        self.rows = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 2:
            # unpacking
            images, targets = batch

            for image, target, pred in zip(
                images,
                targets,
                outputs,
            ):
                box_data_gt = [
                    {
                        "position": {
                            "minX": int(target["boxes"][j][0]),
                            "minY": int(target["boxes"][j][1]),
                            "maxX": int(target["boxes"][j][2]),
                            "maxY": int(target["boxes"][j][3]),
                        },
                        "domain": "pixel",
                        "class_id": int(target["labels"][j] * 10),
                        "class_labels": class_labels,
                    }
                    for j in range(len(target["labels"]))
                ]

                box_data = [
                    {
                        "position": {
                            "minX": int(pred["boxes"][j][0]),
                            "minY": int(pred["boxes"][j][1]),
                            "maxX": int(pred["boxes"][j][2]),
                            "maxY": int(pred["boxes"][j][3]),
                        },
                        "domain": "pixel",
                        "class_id": int(pred["labels"][j]),
                        "box_caption": f"{pred['scores'][j]:0.3f}",
                        "class_labels": class_labels,
                    }
                    for j in range(len(pred["labels"]))
                ]

                self.rows.append(
                    [
                        wandb.Image(
                            image.cpu(),
                            masks={
                                "ground_truth": {
                                    "mask_data": (target["masks"] * target["labels"][:, None, None])
                                    .max(dim=0)
                                    .values.mul(10)
                                    .cpu()
                                    .numpy(),
                                    "class_labels": class_labels,
                                },
                                "predictions": {
                                    "mask_data": (pred["masks"] * pred["labels"][:, None, None])
                                    .max(dim=0)
                                    .values.cpu()
                                    .numpy(),
                                    "class_labels": class_labels,
                                },
                            },
                            boxes={
                                "ground_truth": {"box_data": box_data_gt},
                                "predictions": {"box_data": box_data},
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
