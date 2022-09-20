from pathlib import Path
from threading import Thread

import albumentations as A
import numpy as np
import torchvision.transforms as T

from data.dataset import SyntheticDataset
from utils import RandomPaste

transform = A.Compose(
    [
        A.LongestMaxSize(max_size=1024),
        A.Flip(),
        RandomPaste(5, "/media/disk1/lfainsin/SPHERES/WHITE", "/dev/null"),
        A.ToGray(p=0.01),
        A.ISONoise(),
        A.ImageCompression(),
    ],
)

dataset = SyntheticDataset(image_dir="/media/disk1/lfainsin/BACKGROUND/coco/", transform=transform)
transform = T.ToPILImage()


def render(i, image, mask):
    image = transform(image)
    mask = transform(mask)

    path = f"/media/disk1/lfainsin/TRAIN_prerender/{i:06d}/"
    Path(path).mkdir(parents=True, exist_ok=True)

    image.save(f"{path}/image.jpg")
    mask.save(f"{path}/MASK.PNG")


def renderlist(list_i, dataset):
    for i in list_i:
        image, mask = dataset[i]
        render(i, image, mask)


sublists = np.array_split(range(len(dataset)), 16 * 5)
threads = []
for sublist in sublists:
    t = Thread(target=renderlist, args=(sublist, dataset))
    t.start()
    threads.append(t)

# join all threads
for t in threads:
    t.join()
