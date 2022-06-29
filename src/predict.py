import argparse
import logging

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from unet import UNet


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
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.5,
        help="Minimum probability value to consider a mask pixel white",
    )

    return parser.parse_args()


def predict_img(net, img, device, threshold):
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    net.eval()
    with torch.inference_mode():
        output = net(img)
        preds = torch.sigmoid(output)[0]
        full_mask = preds.cpu().squeeze()

    return np.asarray(full_mask > threshold)


if __name__ == "__main__":
    args = get_args()

    net = UNet(n_channels=3, n_classes=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    logging.info("Transfering model to device")
    net.to(device=device)

    logging.info(f"Loading model {args.model}")
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info(f"Loading image {args.input}")
    img = Image.open(args.input).convert("RGB")

    logging.info(f"Preprocessing image {args.input}")
    tf = A.Compose(
        [
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ],
    )
    aug = tf(image=np.asarray(img))
    img = aug["image"]

    logging.info(f"Predicting image {args.input}")
    mask = predict_img(net=net, img=img, threshold=args.threshold, device=device)

    logging.info(f"Saving prediction to {args.output}")
    mask = Image.fromarray(mask)
    mask.write(args.output)
