import os
from pathlib import Path

import albumentations as A
import numpy as np
import torch
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
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8) // 255

        # convert image & mask to Tensor float in [0, 1]
        post_process = A.Compose(
            [
                A.SmallestMaxSize(1024),
                A.ToFloat(max_value=255),
                ToTensorV2(),
            ],
        )
        augmentations = post_process(image=image, mask=mask)
        image = augmentations["image"]
        mask = augmentations["mask"]

        # make sure image and mask are floats, TODO: mettre dans le post_process, ToFloat Image only
        image = image.float()
        mask = mask.float()

        return image, mask


class LabeledDataset2(Dataset):
    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)

    def __len__(self):
        return len(list(self.image_dir.iterdir()))

    def __getitem__(self, index):
        path = self.image_dir / str(index)

        # open and convert image
        image = np.array(Image.open(path / "image.jpg").convert("RGB"), dtype=np.uint8)

        # open and convert mask
        mask = np.array(Image.open(path / "MASK.PNG").convert("L"), dtype=np.uint8) // 255

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

        # make sure image and mask are floats, TODO: mettre dans le post_process, ToFloat Image only
        image = image.float()
        mask = mask.float()

        return image, mask


class LabeledDataset3(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, idx):
        # create paths from ids
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])

        # load image and mask
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # convert mask to numpy array to apply operations
        mask = np.array(mask)

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]  # first id is the background, so remove it

        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        bboxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            bboxes.append([xmin, ymin, xmax, ymax])

        # convert arrays to tensors
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)  # there is only one class
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # image_id = torch.tensor([idx])
        # area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)  # suppose all instances are not crowd

        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["masks"] = masks
        # target["image_id"] = image_id
        # target["area"] = area
        # target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
