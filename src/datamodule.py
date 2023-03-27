import datasets
import torch
from lightning.pytorch import LightningDataModule
from lightning.pytorch.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import AugMix
from transformers import DetrFeatureExtractor


class DETRDataModule(LightningDataModule):
    """PyTorch Lightning data module for DETR."""

    def __init__(
        self,
        num_workers: int = 8,
        batch_size: int = 6,
        prefetch_factor: int = 2,
        model_name: str = "facebook/detr-resnet-50",
        persistent_workers: bool = True,
    ):
        """Constructor.

        Args:
            num_workers (int, optional): Number of workers.
            batch_size (int, optional): Batch size.
            prefetch_factor (int, optional): Prefetch factor.
            val_split (float, optional): Validation split.
            model_name (str, optional): Model name.
        """
        super().__init__()

        # save params
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

        # get feature extractor
        self.feature_extractor = DetrFeatureExtractor.from_pretrained(model_name)

    def prepare_data(self):
        """Download data and prepare for training."""
        # load datasets
        self.illumination = datasets.load_dataset("src/spheres_illumination.py", split="train")
        self.render = datasets.load_dataset("src/spheres_synth.py", split="train")
        self.real = datasets.load_dataset("src/spheres.py", split="train")

        # split datasets
        self.illumination = self.illumination.train_test_split(test_size=0.01)
        self.render = self.render.train_test_split(test_size=0.01)
        self.real = self.real.train_test_split(test_size=0.1)

        # print some info
        print(f"illumination: {self.illumination}")
        print(f"render: {self.render}")
        print(f"real: {self.real}")

        # other datasets
        self.test_ds = datasets.load_dataset("src/spheres_illumination.py", split="test")
        # self.predict_ds = datasets.load_dataset("src/spheres.py", split="train").shuffle().select(range(16))
        self.predict_ds = datasets.load_dataset("src/spheres_predict.py", split="train")

        # define AugMix transform
        self.mix = AugMix()

        # useful mappings
        self.labels = self.real["test"].features["objects"][0]["category_id"].names
        self.id2label = {k: v for k, v in enumerate(self.labels)}
        self.label2id = {v: k for k, v in enumerate(self.labels)}

    def train_transform(self, batch):
        """Training transform.

        Args:
            batch (dict): Batch precollated by HuggingFace datasets.
            Structure is similar to the following:
            {
                "image": list[PIL.Image],
                "image_id": list[int],
                "objects": [
                    {
                        "bbox": list[float, 4],
                        "category_id": int,
                    }
                ]
            }

        Returns:
            dict: Augmented and processed batch.
            Structure is similar to the following:
            {
                "pixel_values": TensorType["batch", "canal", "width", "height"],
                "pixel_mask": TensorType["batch", 1200, 1200],
                "labels": List[Dict[str, TensorType["batch", "num_boxes", "num_labels"]]],
            }
        """
        # extract images, ids and objects from batch
        images = batch["image"]
        ids = batch["image_id"]
        objects = batch["objects"]

        # apply AugMix transform
        images_mixed = [self.mix(image) for image in images]

        # build targets for feature extractor
        targets = [
            {
                "image_id": id,
                "annotations": object,
            }
            for id, object in zip(ids, objects)
        ]

        # process images and targets with feature extractor for DETR
        processed = self.feature_extractor(
            images=images_mixed,
            annotations=targets,
            return_tensors="pt",
        )

        return processed

    def val_transform(self, batch):
        """Validation transform.

        Just like Training transform, but without AugMix.
        """
        # extract images, ids and objects from batch
        images = batch["image"]
        ids = batch["image_id"]
        objects = batch["objects"]

        # build targets for feature extractor
        targets = [
            {
                "image_id": id,
                "annotations": object,
            }
            for id, object in zip(ids, objects)
        ]

        processed = self.feature_extractor(
            images=images,
            annotations=targets,
            return_tensors="pt",
        )

        return processed

    def predict_transform(self, batch):
        """Prediction transform.

        Just like val_transform, but with images.
        """
        processed = self.val_transform(batch)

        # add images to dict
        processed["images"] = batch["image"]

        return processed

    def collate_fn(self, examples):
        """Collate function.

        Convert list of dicts to dict of Tensors.
        """
        return {
            "pixel_values": torch.stack([data["pixel_values"] for data in examples]),
            "pixel_mask": torch.stack([data["pixel_mask"] for data in examples]),
            "labels": [data["labels"] for data in examples],
        }

    def collate_fn_predict(self, examples):
        """Collate function.

        Convert list of dicts to dict of Tensors.
        """
        return {
            "pixel_values": torch.stack([data["pixel_values"] for data in examples]),
            "pixel_mask": torch.stack([data["pixel_mask"] for data in examples]),
            "labels": [data["labels"] for data in examples],
            "images": [data["images"] for data in examples],
        }

    def train_dataloader(self):
        """Training dataloader."""
        loaders = {
            "illumination": DataLoader(
                self.illumination["train"].with_transform(self.val_transform),
                shuffle=True,
                pin_memory=True,
                persistent_workers=self.persistent_workers,
                collate_fn=self.collate_fn,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
            ),
            "render": DataLoader(
                self.render["train"].with_transform(self.val_transform),
                shuffle=True,
                pin_memory=True,
                persistent_workers=self.persistent_workers,
                collate_fn=self.collate_fn,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
            ),
            "real": DataLoader(
                self.real["train"].with_transform(self.val_transform),
                shuffle=True,
                pin_memory=True,
                persistent_workers=self.persistent_workers,
                collate_fn=self.collate_fn,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
            ),
        }

        return CombinedLoader(loaders, mode="max_size_cycle")

    def val_dataloader(self):
        """Validation dataloader."""
        loaders = {
            "illumination": DataLoader(
                self.illumination["test"].with_transform(self.val_transform),
                pin_memory=True,
                persistent_workers=self.persistent_workers,
                collate_fn=self.collate_fn,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
            ),
            "render": DataLoader(
                self.render["test"].with_transform(self.val_transform),
                pin_memory=True,
                persistent_workers=self.persistent_workers,
                collate_fn=self.collate_fn,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
            ),
            "real": DataLoader(
                self.real["test"].with_transform(self.val_transform),
                pin_memory=True,
                persistent_workers=self.persistent_workers,
                collate_fn=self.collate_fn,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
            ),
        }

        return CombinedLoader(loaders, mode="max_size_cycle")

    def predict_dataloader(self):
        """Prediction dataloader."""
        return DataLoader(
            self.predict_ds.with_transform(self.predict_transform),
            pin_memory=True,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collate_fn_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )


if __name__ == "__main__":
    # load data
    dm = DETRDataModule()
    dm.prepare_data()
    ds = dm.train_dataloader()

    for batch in ds:
        print(batch)
