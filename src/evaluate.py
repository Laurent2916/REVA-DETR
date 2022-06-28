import torch
from tqdm import tqdm

import wandb
from src.utils.dice import dice_coeff


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with tqdm(dataloader, total=len(dataloader.dataset), desc="Validation", unit="img", leave=False) as pbar:
        for images, masks_true in dataloader:
            # move images and labels to correct device
            images = images.to(device=device)
            masks_true = masks_true.unsqueeze(1).to(device=device)

            with torch.inference_mode():
                # predict the mask
                masks_pred = net(images)
                masks_pred = (torch.sigmoid(masks_pred) > 0.5).float()

                # compute the Dice score
                dice_score += dice_coeff(masks_pred, masks_true, reduce_batch_first=False)

            pbar.update(images.shape[0])

    # save some images to wandb
    table = wandb.Table(columns=["image", "mask", "prediction"])
    for img, mask, pred in zip(images.to("cpu"), masks_true.to("cpu"), masks_pred.to("cpu")):
        table.add_data(wandb.Image(img), wandb.Image(mask), wandb.Image(pred))
    wandb.log({"predictions_table": table}, commit=False)

    net.train()

    # Fixes a potential division by zero error
    return dice_score / num_val_batches if num_val_batches else dice_score
