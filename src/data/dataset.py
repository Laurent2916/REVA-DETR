from pathlib import Path

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class SphereDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.images = list(Path(image_dir).glob("**/*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.array(Image.open(self.images[index]).convert("RGB"), dtype=np.uint8)

        if self.transform is not None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        else:
            mask_path = self.images[index].parent.joinpath("MASK.PNG")
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8) / 255

            preprocess = A.Compose(
                [
                    A.SmallestMaxSize(1024),
                    A.ToFloat(max_value=255),
                    ToTensorV2(),
                ],
            )
            augmentations = preprocess(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        # make sure image and mask are floats
        image = image.float()
        mask = mask.float()

        return image, mask
