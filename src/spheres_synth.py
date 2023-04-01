import pathlib

import datasets

dataset_path = pathlib.Path("./dataset_render/")

_VERSION = "1.0.0"

_DESCRIPTION = ""

_HOMEPAGE = ""

_LICENSE = ""

_NAMES = [
    "Matte",
    "Shiny",
    "Chrome",
]


class SphereSynth(datasets.GeneratorBasedBuilder):

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
                    "dataset_path": dataset_path,
                },
            ),
        ]

    def _generate_examples(self, dataset_path: pathlib.Path):
        """Generate images and labels for splits."""
        # create png iterator
        width = 1200
        height = 675
        object_index = 0
        pngs = dataset_path.glob("*.png")
        for index, png in enumerate(pngs):
            # open corresponding csv file
            csv = dataset_path / (png.stem + ".csv")

            # read csv lines
            with open(csv, "r") as f:
                lines = f.readlines()
                lines = [line.strip().split(",") for line in lines]
                lines = [
                    (
                        float(line[0]),
                        1 - float(line[1]),
                        float(line[2]),
                        1 - float(line[3]),
                        line[4].strip()
                    ) for line in lines
                ]

                bboxes = [
                    (
                        line[0] * width,
                        line[3] * height,
                        (line[2] - line[0]) * width,
                        (line[1] - line[3]) * height,
                    )
                    for line in lines
                ]

                categories = []
                for line in lines:
                    category = line[4]

                    if category == "White":
                        category = "Matte"
                    elif category == "Black":
                        category = "Shiny"
                    elif category == "Grey":
                        category = "Matte"
                    elif category == "Red":
                        category = "Shiny"
                    elif category == "Chrome":
                        category = "Chrome"
                    elif category == "Cyan":
                        category = "Shiny"

                    categories.append(category)

            # generate data
            data = {
                "image_id": index,
                "image": str(png),
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
                    for bbox, category in zip(bboxes, categories)
                ],
            }

            yield index, data


if __name__ == "__main__":
    from PIL import ImageDraw

    # load dataset
    dataset = datasets.load_dataset("src/spheres_synth.py", split="train")

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
        image.save(f"example_synth_{idx}.jpg")
