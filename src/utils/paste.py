import random as rd
from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image, ImageEnhance


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
        scale_range=(0.1, 0.2),
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

    def apply(self, img, positions, paste_img, paste_mask, **params):
        # convert img to Image, needed for `paste` function
        img = Image.fromarray(img)

        # paste spheres
        for pos in positions:
            img.paste(paste_img, pos, paste_mask)

        return np.asarray(img.convert("RGB"))

    def apply_to_mask(self, mask, positions, paste_mask, **params):
        # convert mask to Image, needed for `paste` function
        mask = Image.fromarray(mask)

        # binarize the mask -> {0, 1}
        paste_mask_bin = paste_mask.point(lambda p: 1 if p > 10 else 0)

        # paste spheres
        for pos in positions:
            mask.paste(paste_mask, pos, paste_mask_bin)

        return np.asarray(mask.convert("L"))

    @staticmethod
    def overlap(positions, x1, y1, w, h):
        for x2, y2 in positions:
            if x1 + w >= x2 and x1 <= x2 + w and y1 + h >= y2 and y1 <= y2 + h:
                return True
        return False

    def get_params_dependent_on_targets(self, params):
        # choose a random image and its corresponding mask
        img_path = rd.choice(self.images)
        mask_path = img_path.parent.joinpath("MASK.PNG")

        # load the "paste" image
        paste_img = Image.open(img_path).convert("RGBA")

        # load its respective mask
        paste_mask = Image.open(mask_path).convert("LA")

        # load the target image
        target_img = params["image"]

        # compute shapes, for easier computations
        target_shape = np.array(target_img.shape[:2], dtype=np.uint)
        paste_shape = np.array(paste_img.size, dtype=np.uint)

        # change paste_img's contrast randomly
        filter = ImageEnhance.Contrast(paste_img)
        paste_img = filter.enhance(rd.uniform(0.5, 1.5))

        # change paste_img's brightness randomly
        filter = ImageEnhance.Brightness(paste_img)
        paste_img = filter.enhance(rd.uniform(0.5, 1.5))

        # compute the minimum scaling to fit inside target image
        min_scale = np.min(target_shape / paste_shape)

        # randomize the relative scaling
        scale = rd.uniform(*self.scale_range)

        # rotate the image and its mask
        angle = rd.uniform(0, 360)
        paste_img = paste_img.rotate(angle, expand=True)
        paste_mask = paste_mask.rotate(angle, expand=True)

        # scale the "paste" image and its mask
        paste_img = paste_img.resize(
            tuple((paste_shape * min_scale * scale).astype(np.uint)),
            resample=Image.Resampling.LANCZOS,
        )
        paste_mask = paste_mask.resize(
            tuple((paste_shape * min_scale * scale).astype(np.uint)),
            resample=Image.Resampling.LANCZOS,
        )

        # update paste_shape after scaling
        paste_shape = np.array(paste_img.size, dtype=np.uint)

        # generate some positions
        positions = []
        NB = rd.randint(1, self.nb)
        while len(positions) < NB:
            x = rd.randint(0, target_shape[0] - paste_shape[0])
            y = rd.randint(0, target_shape[1] - paste_shape[1])

            # check for overlapping
            if RandomPaste.overlap(positions, x, y, paste_shape[0], paste_shape[1]):
                continue

            positions.append((x, y))

        params.update(
            {
                "positions": positions,
                "paste_img": paste_img,
                "paste_mask": paste_mask,
            }
        )

        return params
