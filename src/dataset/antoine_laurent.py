import json
import pathlib

import cv2
import datasets
import numpy as np

prefix = "/data/local-files/?d=spheres/"
dataset_path = pathlib.Path("./dataset_antoine_laurent/")
annotation_path = dataset_path / "annotations.json"  # from labelstudio

_VERSION = "2.0.0"

_DESCRIPTION = ""

_HOMEPAGE = ""

_LICENSE = ""

_NAMES = [
    "Matte",
    "Shiny",
    "Chrome",
]


class SphereAntoineLaurent(datasets.GeneratorBasedBuilder):
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
                    "annotation_path": annotation_path,
                },
            ),
        ]

    def _generate_examples(self, dataset_path: pathlib.Path, annotation_path: pathlib.Path):
        """Generate images and labels for splits."""
        with open(annotation_path, "r") as f:
            tasks = json.load(f)
            index = 0

            for task in tasks:
                image_id = task["id"]
                image_name = task["data"]["img"]
                image_name = image_name[len(prefix) :]
                image_name = pathlib.Path(image_name)

                # check image_name exists
                assert (dataset_path / image_name).is_file()

                # create annotation groups
                annotation_groups: dict[str, list[dict]] = {}
                for annotation in task["annotations"][0]["result"]:
                    id = annotation["id"]
                    if "parentID" in annotation:
                        parent_id = annotation["parentID"]
                        if parent_id not in annotation_groups:
                            annotation_groups[parent_id] = []
                        annotation_groups[parent_id].append(annotation)
                    else:
                        if id not in annotation_groups:
                            annotation_groups[id] = []
                        annotation_groups[id].append(annotation)

                # check all annotations have same width and height
                width = task["annotations"][0]["result"][0]["original_width"]
                height = task["annotations"][0]["result"][0]["original_height"]
                for annotation in task["annotations"][0]["result"]:
                    assert annotation["original_width"] == width
                    assert annotation["original_height"] == height

                # check all childs of group have same label
                labels = {}
                for group_id, annotations in annotation_groups.items():
                    label = annotations[0]["value"]["keypointlabels"][0]
                    for annotation in annotations:
                        assert annotation["value"]["keypointlabels"][0] == label

                    # convert labels
                    if label == "White":
                        label = "Matte"
                    elif label == "Black":
                        label = "Shiny"
                    elif label == "Red":
                        label = "Shiny"

                    labels[group_id] = label

                # compute bboxes
                bboxes = {}
                for group_id, annotations in annotation_groups.items():
                    # convert points to numpy array
                    points = np.array(
                        [
                            [
                                annotation["value"]["x"] / 100 * width,
                                annotation["value"]["y"] / 100 * height,
                            ]
                            for annotation in annotations
                        ],
                        dtype=np.float32,
                    )

                    # fit ellipse from points
                    ellipse = cv2.fitEllipse(points)

                    # extract ellipse parameters
                    x_C = ellipse[0][0]
                    y_C = ellipse[0][1]
                    a = ellipse[1][0] / 2
                    b = ellipse[1][1] / 2
                    theta = ellipse[2] * np.pi / 180

                    # sample ellipse points
                    t = np.linspace(0, 2 * np.pi, 100)
                    x = x_C + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
                    y = y_C + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)

                    # get bounding box
                    xmin = np.min(x)
                    xmax = np.max(x)
                    ymin = np.min(y)
                    ymax = np.max(y)

                    w = xmax - xmin
                    h = ymax - ymin

                    # bboxe to coco format
                    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/detr/image_processing_detr.py#L295
                    bboxes[group_id] = [xmin, ymin, w, h]

                # compute areas
                areas = {group_id: w * h for group_id, (_, _, w, h) in bboxes.items()}

                # generate data
                data = {
                    "image_id": image_id,
                    "image": str(dataset_path / image_name),
                    "width": width,
                    "height": height,
                    "objects": [
                        {
                            # "category_id": "White",
                            "category_id": labels[group_id],
                            "image_id": image_id,
                            "id": group_id,
                            "area": areas[group_id],
                            "bbox": bboxes[group_id],
                            "iscrowd": False,
                        }
                        for group_id in annotation_groups
                    ],
                }

                yield index, data
                index += 1


if __name__ == "__main__":
    from PIL import ImageDraw

    # load dataset
    dataset = datasets.load_dataset("src/spheres.py", split="train")
    print("dataset loaded")

    labels = dataset.features["objects"][0]["category_id"].names
    id2label = {k: v for k, v in enumerate(labels)}
    label2id = {v: k for k, v in enumerate(labels)}

    print(f"labels: {labels}")
    print(f"id2label: {id2label}")
    print(f"label2id: {label2id}")
    print()

    idx = 0
    while True:
        image = dataset[idx]["image"]
        if "DSC_4234" in image.filename:
            break
        idx += 1

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
    image.save("example_antoine_laurent.jpg")
