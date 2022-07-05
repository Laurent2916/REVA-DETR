import argparse
import logging

import albumentations as A
import cv2
import numpy as np
import torch
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

    net = cv2.dnn.readNetFromONNX(args.model)
    logging.info("onnx model loaded")

    logging.info(f"Loading image {args.input}")
    input_img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    input_img = input_img.astype(np.float32)
    # input_img = cv2.resize(input_img, (512, 512))

    logging.info("converting to blob")
    input_blob = cv2.dnn.blobFromImage(
        image=input_img,
        scalefactor=1 / 255,
    )

    net.setInput(input_blob)
    mask = net.forward()
    mask = sigmoid(mask)
    mask = mask > 0.5
    mask = mask.astype(np.float32)

    logging.info(f"Saving prediction to {args.output}")
    mask = Image.fromarray(mask, "L")
    mask.save(args.output)
