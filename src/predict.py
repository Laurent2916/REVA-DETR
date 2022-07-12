import argparse
import logging

import albumentations as A
import numpy as np
import onnx
import onnxruntime
from albumentations.pytorch import ToTensorV2
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser(
        description="Predict masks from input images",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="model.pth",
        metavar="FILE",
        help="Specify the file in which the model is stored",
    )
    parser.add_argument(
        "--input",
        "-i",
        metavar="INPUT",
        help="Filenames of input images",
        required=True,
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="OUTPUT",
        help="Filenames of output images",
    )

    return parser.parse_args()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    onnx_model = onnx.load(args.model)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(args.model)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    img = Image.open(args.input).convert("RGB")

    logging.info(f"Preprocessing image {args.input}")
    transform = A.Compose(
        [
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ],
    )
    aug = transform(image=np.asarray(img))
    img = aug["image"]

    logging.info(f"Predicting image {args.input}")
    img = img.unsqueeze(0)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
    ort_outs = ort_session.run(None, ort_inputs)

    img_out_y = ort_outs[0]

    img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode="L")

    img_out_y.save(args.output)
