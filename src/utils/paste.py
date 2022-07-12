import random as rd
from pathlib import Path

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
        image_dir,
        scale_range=(0.05, 0.5),
        always_apply=True,
        p=1.0,
    ):
        super().__init__(always_apply, p)
        self.images = []
        self.images.extend(list(Path(image_dir).glob("**/*.jpg")))
        self.images.extend(list(Path(image_dir).glob("**/*.png")))
        self.scale_range = scale_range
        self.nb = nb

    @property
    def targets_as_params(self):
        return ["image"]

    def apply(self, img, augmentations, paste_img, paste_mask, **params):
        # convert img to Image, needed for `paste` function
        img = Image.fromarray(img)

        # copy paste_img and paste_mask
        paste_mask = paste_mask.copy()
        paste_img = paste_img.copy()

        # paste spheres
        for (x, y, shearx, sheary, shape, angle, brightness, contrast) in augmentations:
            paste_img = T.functional.adjust_contrast(
                paste_img,
                contrast_factor=contrast,
            )
            paste_img = T.functional.adjust_brightness(
                paste_img,
                brightness_factor=brightness,
            )
            paste_img = T.functional.affine(
                paste_img,
                scale=0.95,
                angle=angle,
                translate=(0, 0),
                shear=(shearx, sheary),
                interpolation=T.InterpolationMode.BILINEAR,
            )
            paste_img = T.functional.resize(
                paste_img,
                size=shape,
                interpolation=T.InterpolationMode.BILINEAR,
            )

            paste_mask = T.functional.affine(
                paste_mask,
                scale=0.95,
                angle=angle,
                translate=(0, 0),
                shear=(shearx, sheary),
                interpolation=T.InterpolationMode.BILINEAR,
            )
            paste_mask = T.functional.resize(
                paste_mask,
                size=shape,
                interpolation=T.InterpolationMode.BILINEAR,
            )

            img.paste(paste_img, (x, y), paste_mask)

        return np.array(img.convert("RGB"))

    def apply_to_mask(self, mask, augmentations, paste_mask, **params):
        # convert mask to Image, needed for `paste` function
        mask = Image.fromarray(mask)

        # copy paste_img and paste_mask
        paste_mask = paste_mask.copy()

        for (x, y, shearx, sheary, shape, angle, _, _) in augmentations:
            paste_mask = T.functional.affine(
                paste_mask,
                scale=0.95,
                angle=angle,
                translate=(0, 0),
                shear=(shearx, sheary),
                interpolation=T.InterpolationMode.BILINEAR,
            )
            paste_mask = T.functional.resize(
                paste_mask,
                size=shape,
                interpolation=T.InterpolationMode.BILINEAR,
            )

            # binarize the mask -> {0, 1}
            paste_mask_bin = paste_mask.point(lambda p: 1 if p > 10 else 0)

            mask.paste(paste_mask, (x, y), paste_mask_bin)

        return np.array(mask.convert("L"))

    def get_params_dependent_on_targets(self, params):
        # choose a random image and its corresponding mask
        img_path = rd.choice(self.images)
        mask_path = img_path.parent.joinpath("MASK.PNG")

        # load images (w/ transparency)
        paste_img = Image.open(img_path).convert("RGBA")
        paste_mask = Image.open(mask_path).convert("LA")
        target_img = params["image"]

        # compute shapes
        target_shape = np.array(target_img.shape[:2], dtype=np.uint)
        paste_shape = np.array(paste_img.size, dtype=np.uint)

        # compute minimum scaling to fit inside target
        min_scale = np.min(target_shape / paste_shape)

        # generate augmentations
        augmentations = []
        NB = rd.randint(1, self.nb)
        ite = 0
        while len(augmentations) < NB:

            if ite > 100:
                break

            scale = rd.uniform(*self.scale_range) * min_scale
            shape = np.array(paste_shape * scale, dtype=np.uint)

            x = rd.randint(0, target_shape[0] - shape[0])
            y = rd.randint(0, target_shape[1] - shape[1])

            # check for overlapping
            if RandomPaste.overlap(augmentations, x, y, shape[0], shape[1]):
                continue

            shearx = rd.uniform(-2, 2)
            sheary = rd.uniform(-2, 2)

            angle = rd.uniform(0, 360)

            brightness = rd.uniform(0.8, 1.2)
            contrast = rd.uniform(0.8, 1.2)

            augmentations.append((x, y, shearx, sheary, tuple(shape), angle, brightness, contrast))

        params.update(
            {
                "augmentations": augmentations,
                "paste_img": paste_img,
                "paste_mask": paste_mask,
            }
        )

        return params

    @staticmethod
    def overlap(positions, x1, y1, w, h):
        for x2, y2, _, _, _, _, _, _ in positions:
            if x1 + w >= x2 and x1 <= x2 + w and y1 + h >= y2 and y1 <= y2 + h:
                return True
        return False
