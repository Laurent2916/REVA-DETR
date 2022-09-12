import os
from pathlib import Path

import albumentations as A
import numpy as np
import torch
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
        image = np.ascontiguousarray(
            Image.open(
                self.images[index],
            ).convert("RGB"),
            dtype=np.uint8,
        )

        # create empty mask of same size
        mask = np.zeros(
            (*image.shape[:2], 4),
            dtype=np.uint8,
        )

        # augment image and mask
        augmentations = self.transform(image=image, mask=mask)
        image = augmentations["image"]
        mask = augmentations["mask"]

        return image, mask


class RealDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms

        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

        self.res = A.LongestMaxSize(max_size=1024)

    def __getitem__(self, idx):
        # create paths from ids
        image_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])

        # load image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # convert to numpy arrays
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        # resize images, TODO: remove ?
        aug = self.res(image=image, mask=mask)
        image = aug["image"]
        mask = aug["mask"]

        # get ids from mask
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]  # first id is the background, so remove it

        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]
        masks = masks.astype(np.uint8)  # cast to uint8 for albumentations

        # create bboxes from masks (pascal format)
        num_objs = len(obj_ids)
        bboxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            bboxes.append([xmin, ymin, xmax, ymax])

        # convert arrays for albumentations
        bboxes = torch.as_tensor(bboxes, dtype=torch.int64)
        labels = torch.ones((num_objs,), dtype=torch.int64)  # assume there is only one class (id=1)
        masks = list(np.asarray(masks))

        if self.transforms is not None:
            # arrange transform data
            data = {
                "image": image,
                "labels": labels,
                "bboxes": bboxes,
                "masks": masks,
            }
            # apply transform
            augmented = self.transforms(**data)
            # get augmented data
            image = augmented["image"]
            bboxes = augmented["bboxes"]
            labels = augmented["labels"]
            masks = augmented["masks"]

        bboxes = torch.as_tensor(bboxes, dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)  # int64 required by torchvision maskrcnn
        masks = torch.stack(masks)  # stack masks, required by torchvision maskrcnn

        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)  # assume all instances are not crowd

        target = {
            "boxes": bboxes,
            "labels": labels,
            "masks": masks,
            "area": area,
            "image_id": image_id,
            "iscrowd": iscrowd,
        }

        return image, target

    def __len__(self):
        return len(self.imgs)


class LabeledDataset(Dataset):
    def __init__(self, image_dir, transforms):
        self.images = list(Path(image_dir).glob("**/*.jpg"))
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # open and convert image
        image = np.ascontiguousarray(
            Image.open(self.images[idx]).convert("RGB"),
        )

        # open and convert mask
        mask_path = self.images[idx].parent.joinpath("MASK.PNG")
        mask = np.ascontiguousarray(
            Image.open(mask_path).convert("L"),
        )

        # get ids from mask
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]  # first id is the background, so remove it

        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]
        masks = masks.astype(np.uint8)  # cast to uint8 for albumentations

        # create bboxes from masks (pascal format)
        num_objs = len(obj_ids)
        bboxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            bboxes.append([xmin, ymin, xmax, ymax])

        # convert arrays for albumentations
        bboxes = torch.as_tensor(bboxes, dtype=torch.int64)
        labels = torch.ones((num_objs,), dtype=torch.int64)  # assume there is only one class (id=1)
        masks = list(np.asarray(masks))

        if self.transforms is not None:
            # arrange transform data
            data = {
                "image": image,
                "labels": labels,
                "bboxes": bboxes,
                "masks": masks,
            }
            # apply transform
            augmented = self.transforms(**data)
            # get augmented data
            image = augmented["image"]
            bboxes = augmented["bboxes"]
            labels = augmented["labels"]
            masks = augmented["masks"]

        bboxes = torch.as_tensor(bboxes, dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)  # int64 required by torchvision maskrcnn
        masks = torch.stack(masks)  # stack masks, required by torchvision maskrcnn

        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)  # assume all instances are not crowd

        target = {
            "boxes": bboxes,
            "labels": labels,
            "masks": masks,
            "area": area,
            "image_id": image_id,
            "iscrowd": iscrowd,
        }

        return image, target
