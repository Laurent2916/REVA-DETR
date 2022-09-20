from __future__ import annotations

import random as rd
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import numpy as np
import torchvision.transforms as T
from PIL import Image


class RandomPaste(A.DualTransform):
    """Paste an object on a background.

    Args:
        TODO

    Targets:
        image, mask

    Image types:
        uint8
    """

    def __init__(
        self,
        nb,
        sphere_image_dir,
        chrome_sphere_image_dir,
        scale_range=(0.05, 0.3),
        always_apply=True,
        p=1.0,
    ):
        super().__init__(always_apply, p)

        self.sphere_images = []
        self.sphere_images.extend(list(Path(sphere_image_dir).glob("**/*.jpg")))
        self.sphere_images.extend(list(Path(sphere_image_dir).glob("**/*.png")))

        self.chrome_sphere_images = []
        self.chrome_sphere_images.extend(list(Path(chrome_sphere_image_dir).glob("**/*.jpg")))
        self.chrome_sphere_images.extend(list(Path(chrome_sphere_image_dir).glob("**/*.png")))

        self.scale_range = scale_range
        self.nb = nb

    @property
    def targets_as_params(self):
        return ["image"]

    def apply(self, img, augmentation_datas, **params):
        # convert img to Image, needed for `paste` function
        img = Image.fromarray(img)

        # paste spheres
        for augmentation in augmentation_datas:
            paste_img_aug = T.functional.adjust_contrast(
                augmentation.paste_img,
                contrast_factor=augmentation.contrast,
            )
            paste_img_aug = T.functional.adjust_brightness(
                paste_img_aug,
                brightness_factor=augmentation.brightness,
            )
            paste_img_aug = T.functional.affine(
                paste_img_aug,
                scale=0.95,
                translate=(0, 0),
                angle=augmentation.angle,
                shear=augmentation.shear,
                interpolation=T.InterpolationMode.BICUBIC,
            )
            paste_img_aug = T.functional.resize(
                paste_img_aug,
                size=augmentation.shape,
                interpolation=T.InterpolationMode.LANCZOS,
            )

            paste_mask_aug = T.functional.affine(
                augmentation.paste_mask,
                scale=0.95,
                translate=(0, 0),
                angle=augmentation.angle,
                shear=augmentation.shear,
                interpolation=T.InterpolationMode.BICUBIC,
            )
            paste_mask_aug = T.functional.resize(
                paste_mask_aug,
                size=augmentation.shape,
                interpolation=T.InterpolationMode.LANCZOS,
            )

            img.paste(paste_img_aug, augmentation.position, paste_mask_aug)

        return np.array(img.convert("RGB"))

    def apply_to_mask(self, mask, augmentation_datas, **params):
        # convert mask to Image, needed for `paste` function
        mask = Image.fromarray(mask)

        for augmentation in augmentation_datas:
            paste_mask_aug = T.functional.affine(
                augmentation.paste_mask,
                scale=0.95,
                translate=(0, 0),
                angle=augmentation.angle,
                shear=augmentation.shear,
                interpolation=T.InterpolationMode.BICUBIC,
            )
            paste_mask_aug = T.functional.resize(
                paste_mask_aug,
                size=augmentation.shape,
                interpolation=T.InterpolationMode.LANCZOS,
            )

            # binarize the mask -> {0, 1}
            paste_mask_aug_bin = paste_mask_aug.point(lambda p: augmentation.value if p > 10 else 0)

            mask.paste(paste_mask_aug, augmentation.position, paste_mask_aug_bin)

        return np.array(mask.convert("L"))

    def get_params_dependent_on_targets(self, params):
        # init augmentation list
        augmentation_datas: List[AugmentationData] = []

        # load target image (w/ transparency)
        target_img = params["image"]
        target_shape = np.array(target_img.shape[:2], dtype=np.uint)

        # generate augmentations
        ite = 0
        NB = rd.randint(1, self.nb)
        while len(augmentation_datas) < NB:
            if ite > 100:
                break
            else:
                ite += 1

            # choose a random sphere image and its corresponding mask
            if rd.random() > 0.5 or len(self.chrome_sphere_images) == 0:
                img_path = rd.choice(self.sphere_images)
                value = len(augmentation_datas) + 1
            else:
                img_path = rd.choice(self.chrome_sphere_images)
                value = 255 - len(augmentation_datas)
            mask_path = img_path.parent.joinpath("MASK.PNG")

            # load paste assets
            paste_img = Image.open(img_path).convert("RGBA")
            paste_shape = np.array(paste_img.size, dtype=np.uint)
            paste_mask = Image.open(mask_path).convert("LA")

            # compute minimum scaling to fit inside target
            min_scale = np.min(target_shape / paste_shape)

            # randomly scale image inside target
            scale = rd.uniform(*self.scale_range) * min_scale
            shape = np.array(paste_shape * scale, dtype=np.uint)

            try:
                augmentation_datas.append(
                    AugmentationData(
                        position=(
                            rd.randint(0, target_shape[1] - shape[1]),
                            rd.randint(0, target_shape[0] - shape[0]),
                        ),
                        shear=(
                            rd.uniform(-2, 2),
                            rd.uniform(-2, 2),
                        ),
                        shape=tuple(shape),
                        angle=rd.uniform(0, 360),
                        brightness=rd.uniform(0.8, 1.2),
                        contrast=rd.uniform(0.8, 1.2),
                        paste_img=paste_img,
                        paste_mask=paste_mask,
                        value=value,
                        target_shape=tuple(target_shape),
                        other_augmentations=augmentation_datas,
                    )
                )
            except ValueError:
                continue

        params.update(
            {
                "augmentation_datas": augmentation_datas,
            }
        )

        return params


@dataclass
class AugmentationData:
    """Store data for pasting augmentation."""

    position: Tuple[int, int]

    shape: Tuple[int, int]
    target_shape: Tuple[int, int]
    angle: float

    brightness: float
    contrast: float

    shear: Tuple[float, float]

    paste_img: Image.Image
    paste_mask: Image.Image
    value: int

    other_augmentations: List[AugmentationData]

    def __post_init__(self) -> None:
        # check for overlapping
        if overlap(self.other_augmentations, self):
            raise ValueError


def overlap(augmentations: List[AugmentationData], augmentation: AugmentationData) -> bool:
    x1, y1 = augmentation.position
    w1, h1 = augmentation.shape

    for other_augmentation in augmentations:
        x2, y2 = other_augmentation.position
        w2, h2 = other_augmentation.shape

        if x1 + w1 >= x2 and x1 <= x2 + w2 and y1 + h1 >= y2 and y1 <= y2 + h2:
            return True

    return False
