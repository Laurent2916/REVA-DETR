import argparse
import logging
from pathlib import Path

import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from evaluate import evaluate
from src.utils.dataset import SphereDataset
from src.utils.dice import dice_loss
from unet import UNet
from utils.paste import RandomPaste

CHECKPOINT_DIR = Path("./checkpoints/")
DIR_TRAIN_IMG = Path("/home/lilian/data_disk/lfainsin/train2017")
DIR_VALID_IMG = Path("/home/lilian/data_disk/lfainsin/val2017/")
# DIR_VALID_MASK = Path("/home/lilian/data_disk/lfainsin/val2017mask/")
DIR_SPHERE_IMG = Path("/home/lilian/data_disk/lfainsin/spheres/Images/")
DIR_SPHERE_MASK = Path("/home/lilian/data_disk/lfainsin/spheres/Masks/")


def train_net(
    net,
    device,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    save_checkpoint: bool = True,
    amp: bool = False,
):
    # 1. Create transforms
    tf_train = A.Compose(
        [
            A.Flip(),
            A.ColorJitter(),
            RandomPaste(5, 0.2, DIR_SPHERE_IMG, DIR_SPHERE_MASK),
            A.ISONoise(),
            A.ToFloat(max_value=255),
            A.pytorch.ToTensorV2(),
        ],
    )

    tf_valid = A.Compose(
        [
            RandomPaste(5, 0.2, DIR_SPHERE_IMG, DIR_SPHERE_MASK),
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ],
    )

    # 2. Create datasets
    ds_train = SphereDataset(images_dir=DIR_TRAIN_IMG, transform=tf_train)
    # ds_valid = SphereDataset(images_dir=DIR_VALID_IMG, masks_dir=DIR_VALID_MASK, transform=tf_valid)
    ds_valid = SphereDataset(images_dir=DIR_VALID_IMG, transform=tf_valid)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(ds_train, shuffle=True, **loader_args)
    val_loader = DataLoader(ds_valid, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(
        project="U-Net",
        config=dict(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            save_checkpoint=save_checkpoint,
            amp=amp,
        ),
    )

    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(ds_train)}
        Validation size: {len(ds_valid)}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
        """
    )

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0

        with tqdm(total=len(ds_train), desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
            for batch in train_loader:
                images = batch["image"]
                true_masks = batch["mask"]

                assert images.shape[1] == net.n_channels, (
                    f"Network has been defined with {net.n_channels} input channels, "
                    f"but loaded images have {images.shape[1]} channels. Please check that "
                    "the images are loaded correctly."
                )

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) + dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True,
                    )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({"train loss": loss.item(), "step": global_step, "epoch": epoch})
                pbar.set_postfix(**{"loss (batch)": loss.item()})

                # Evaluation round
                division_step = len(ds_train) // (10 * batch_size)
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace("/", ".")
                            histograms["Weights/" + tag] = wandb.Histogram(value.data.cpu())
                            histograms["Gradients/" + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device)
                        scheduler.step(val_score)

                        logging.info("Validation Dice score: {}".format(val_score))
                        experiment.log(
                            {
                                "learning rate": optimizer.param_groups[0]["lr"],
                                "validation Dice": val_score,
                                "images": wandb.Image(images[0].cpu()),
                                "masks": {
                                    "true": wandb.Image(true_masks[0].float().cpu()),
                                    "pred": wandb.Image(
                                        torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()
                                    ),
                                },
                                "step": global_step,
                                "epoch": epoch,
                                **histograms,
                            }
                        )

        if save_checkpoint:
            Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(CHECKPOINT_DIR / "checkpoint_epoch{}.pth".format(epoch)))
            logging.info(f"Checkpoint {epoch} saved!")


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        metavar="E",
        type=int,
        default=5,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        metavar="B",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        "-l",
        metavar="LR",
        type=float,
        default=1e-5,
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--load",
        "-f",
        type=str,
        default=False,
        help="Load model from a .pth file",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Use mixed precision",
    )
    parser.add_argument(
        "--classes",
        "-c",
        type=int,
        default=1,
        help="Number of classes",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    net = UNet(n_channels=3, n_classes=args.classes)

    logging.info(
        f"""Network:
        \t{net.n_channels} input channels
        \t{net.n_classes} output channels (classes)
        """
    )

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f"Model loaded from {args.load}")

    net.to(device=device)

    try:
        train_net(
            net=net,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            amp=args.amp,
        )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), "INTERRUPTED.pth")
        logging.info("Saved interrupt")
        raise
