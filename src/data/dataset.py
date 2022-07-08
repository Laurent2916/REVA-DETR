from pathlib import Path

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.images = list(Path(image_dir).glob("**/*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # open and convert image
        image = np.array(Image.open(self.images[index]).convert("RGB"), dtype=np.uint8)

        # create empty mask of same size
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # augment image and mask
        augmentations = self.transform(image=image, mask=mask)
        image = augmentations["image"]
        mask = augmentations["mask"]

        # convert image & mask to Tensor float in [0, 1]
        post_process = A.Compose(
            [
                A.ToFloat(max_value=255),
                ToTensorV2(),
            ],
        )
        augmentations = post_process(image=image, mask=mask)
        image = augmentations["image"]
        mask = augmentations["mask"]

        # make sure image and mask are floats
        image = image.float()
        mask = mask.float()

        return image, mask


class LabeledDataset(Dataset):
    def __init__(self, image_dir):
        self.images = list(Path(image_dir).glob("**/*.jpg"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # open and convert image
        image = np.array(Image.open(self.images[index]).convert("RGB"), dtype=np.uint8)

        # open and convert mask
        mask_path = self.images[index].parent.joinpath("MASK.PNG")
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8) / 255

        # convert image & mask to Tensor float in [0, 1]
        post_process = A.Compose(
            [
                # A.SmallestMaxSize(1024),
                A.ToFloat(max_value=255),
                ToTensorV2(),
            ],
        )
        augmentations = post_process(image=image, mask=mask)
        image = augmentations["image"]
        mask = augmentations["mask"]

        return image, mask
