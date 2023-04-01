import json
import pathlib

import datasets

dataset_path_train = pathlib.Path("./dataset_illumination/")

_VERSION = "1.0.0"

_DESCRIPTION = ""

_HOMEPAGE = ""

_LICENSE = ""

_NAMES = [
    "Matte",
    "Shiny",
    "Chrome",
]


class SphereIllumination(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            version=_VERSION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            features=datasets.Features(
                {
                    "image_id": datasets.Value("int64"),
                    "image": datasets.Image(),
                    "width": datasets.Value("int32"),
                    "height": datasets.Value("int32"),
                    "objects": [
                        {
                            "category_id": datasets.ClassLabel(names=_NAMES),
                            "image_id": datasets.Value("int64"),
                            "id": datasets.Value("string"),
                            "area": datasets.Value("float32"),
                            "bbox": datasets.Sequence(datasets.Value("float32"), length=4),
                            "iscrowd": datasets.Value("bool"),
                        }
                    ],
                }
            ),
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "dataset_path": dataset_path_train,
                },
            ),
        ]

    def _generate_examples(self, dataset_path: pathlib.Path):
        """Generate images and labels for splits."""
        width = 1500
        height = 1000

        original_width = 6020
        original_height = 4024

        # create png iterator
        object_index = 0
        jpgs = dataset_path.rglob("*.jpg")
        for index, jpg in enumerate(jpgs):

            # filter out probe images
            if "probes" in jpg.parts:
                continue

            # filter out thumbnails
            if "thumb" in jpg.stem:
                continue

            # open corresponding csv file
            json_file = jpg.parent / "meta.json"

            # read json
            with open(json_file, "r") as f:
                meta = json.load(f)

                gray = (
                    (
                        meta["gray"]["bounding_box"]["x"] / original_width * width,
                        meta["gray"]["bounding_box"]["y"] / original_height * height,
                        meta["gray"]["bounding_box"]["w"] / original_width * width,
                        meta["gray"]["bounding_box"]["h"] / original_height * height
                    ),
                    "Matte"
                )

                chrome = (
                    (
                        meta["chrome"]["bounding_box"]["x"] / original_width * width,
                        meta["chrome"]["bounding_box"]["y"] / original_height * height,
                        meta["chrome"]["bounding_box"]["w"] / original_width * width,
                        meta["chrome"]["bounding_box"]["h"] / original_height * height
                    ),
                    "Chrome"
                )

            # generate data
            data = {
                "image_id": index,
                "image": str(jpg),
                "width": width,
                "height": height,
                "objects": [
                    {
                        "category_id": category,
                        "image_id": index,
                        "id": (object_index := object_index + 1),
                        "area": bbox[2] * bbox[3],
                        "bbox": bbox,
                        "iscrowd": False,
                    }
                    for bbox, category in [gray, chrome]
                ],
            }

            yield index, data


if __name__ == "__main__":
    from PIL import ImageDraw

    # load dataset
    dataset = datasets.load_dataset("src/spheres_illumination.py", split="train")
    dataset = dataset.shuffle()

    labels = dataset.features["objects"][0]["category_id"].names
    id2label = {k: v for k, v in enumerate(labels)}
    label2id = {v: k for k, v in enumerate(labels)}

    print(f"labels: {labels}")
    print(f"id2label: {id2label}")
    print(f"label2id: {label2id}")
    print()

    for idx in range(10):
        image = dataset[idx]["image"]

        print(f"image path: {image.filename}")
        print(f"data: {dataset[idx]}")

        draw = ImageDraw.Draw(image)
        for obj in dataset[idx]["objects"]:
            bbox = (
                obj["bbox"][0],
                obj["bbox"][1],
                obj["bbox"][0] + obj["bbox"][2],
                obj["bbox"][1] + obj["bbox"][3],
            )
            draw.rectangle(bbox, outline="red", width=3)
            draw.text(bbox[:2], text=id2label[obj["category_id"]], fill="black")

        # save image
        image.save(f"example_illumination_{idx}.jpg")
