from pathlib import Path

import numpy as np
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

        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        # make sure image and mask are floats
        image = image.float()
        mask = mask.float()

        return image, mask
