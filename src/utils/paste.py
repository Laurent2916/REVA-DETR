import os
import random as rd

import albumentations as A
import cv2
import numpy as np
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
        scale_limit,
        path_paste_img_dir,
        path_paste_mask_dir,
        always_apply=True,
        p=1.0,
    ):
        super().__init__(always_apply, p)
        self.path_paste_img_dir = path_paste_img_dir
        self.path_paste_mask_dir = path_paste_mask_dir
        self.scale_limit = scale_limit
        self.nb = nb

    @property
    def targets_as_params(self):
        return ["image"]

    def apply(self, img, positions, paste_img, paste_mask, **params):
        img = img.copy()

        w, h = paste_mask.shape
        mask_b = paste_mask > 0
        mask_rgb_b = np.stack([mask_b, mask_b, mask_b], axis=2)

        for (x, y) in positions:
            img[x : x + w, y : y + h] = img[x : x + w, y : y + h] * ~mask_rgb_b + paste_img * mask_rgb_b

        return img

    def apply_to_mask(self, mask, positions, paste_mask, **params):
        mask = mask.copy()

        w, h = paste_mask.shape
        mask_b = paste_mask > 0

        for (x, y) in positions:
            mask[x : x + w, y : y + h] = mask[x : x + w, y : y + h] * ~mask_b + mask_b

        return mask

    def get_params_dependent_on_targets(self, params):
        filename = rd.choice(os.listdir(self.path_paste_img_dir))

        paste_img = np.array(
            Image.open(
                os.path.join(
                    self.path_paste_img_dir,
                    filename,
                )
            ).convert("RGB"),
            dtype=np.uint8,
        )

        paste_mask = (
            np.array(
                Image.open(
                    os.path.join(
                        self.path_paste_mask_dir,
                        filename,
                    )
                ).convert("L"),
                dtype=np.float32,
            )
            / 255
        )

        target_img = params["image"]

        min_scale = min(
            target_img.shape[0] / paste_img.shape[0],
            target_img.shape[1] / paste_img.shape[1],
        )

        rescale_rotate = A.Compose(
            [
                A.Rotate(limit=360, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
                A.RandomScale(scale_limit=(min_scale * self.scale_limit - 1, -0.99), always_apply=True),
            ],
        )

        augmentations = rescale_rotate(image=paste_img, mask=paste_mask)
        paste_img = augmentations["image"]
        paste_mask = augmentations["mask"]

        positions = []
        for _ in range(rd.randint(1, self.nb)):
            x = rd.randint(0, target_img.shape[0] - paste_img.shape[0])
            y = rd.randint(0, target_img.shape[1] - paste_img.shape[1])
            positions.append((x, y))

        params.update(
            {
                "positions": positions,
                "paste_img": paste_img,
                "paste_mask": paste_mask,
            }
        )

        return params

    def get_transform_init_args_names(self):
        return "scale_limit", "path_paste_img_dir", "path_paste_mask_dir"
