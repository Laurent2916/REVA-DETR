import numpy as np
import torch
from tqdm import tqdm

import wandb
from src.utils.dice import dice_coeff

class_labels = {
    1: "sphere",
}


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with tqdm(dataloader, total=len(dataloader.dataset), desc="val", unit="img", leave=False) as pbar:
        for images, masks_true in dataloader:
            # move images and labels to correct device
            images = images.to(device=device)
            masks_true = masks_true.unsqueeze(1).float().to(device=device)

            # forward, predict the mask
            with torch.inference_mode():
                masks_pred = net(images)
                masks_pred_bin = (torch.sigmoid(masks_pred) > 0.5).float()

                # compute the Dice score
                dice_score += dice_coeff(masks_pred_bin, masks_true, reduce_batch_first=False)

            # update progress bar
            pbar.update(images.shape[0])

    # save some images to wandb
    table = wandb.Table(columns=["ID", "image", "ground truth", "prediction"])
    for i, (img, mask, pred) in enumerate(zip(images.to("cpu"), masks_true.to("cpu"), masks_pred.to("cpu"))):
        table.add_data(i, wandb.Image(img), wandb.Image(mask), wandb.Image(pred))
    wandb.log({"predictions_table": table})

    net.train()

    # Fixes a potential division by zero error
    return dice_score / num_val_batches if num_val_batches else dice_score
