import argparse
import logging
from pathlib import Path

import albumentations as A
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from evaluate import evaluate
from src.utils.dataset import SphereDataset
from unet import UNet
from utils.paste import RandomPaste

CHECKPOINT_DIR = Path("./checkpoints/")
DIR_TRAIN_IMG = Path("/home/lilian/data_disk/lfainsin/val2017")
DIR_VALID_IMG = Path("/home/lilian/data_disk/lfainsin/val2017/")
DIR_SPHERE_IMG = Path("/home/lilian/data_disk/lfainsin/spheres/Images/")
DIR_SPHERE_MASK = Path("/home/lilian/data_disk/lfainsin/spheres/Masks/")


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
        default=16,
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


def main():
    # get args from cli
    args = get_args()

    # setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # enable cuda, if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # 0. Create network
    net = UNet(n_channels=3, n_classes=args.classes)
    logging.info(
        f"""Network:
        input channels:  {net.n_channels}
        output channels: {net.n_classes}
        """
    )

    # Load weights, if needed
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f"Model loaded from {args.load}")

    # transfer network to device
    net.to(device=device)

    # 1. Create transforms
    tf_train = A.Compose(
        [
            A.Resize(500, 500),
            A.Flip(),
            A.ColorJitter(),
            RandomPaste(5, DIR_SPHERE_IMG, DIR_SPHERE_MASK),
            A.ISONoise(),
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ],
    )
    tf_valid = A.Compose(
        [
            A.Resize(500, 500),
            RandomPaste(5, DIR_SPHERE_IMG, DIR_SPHERE_MASK),
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ],
    )

    # 2. Create datasets
    ds_train = SphereDataset(image_dir=DIR_TRAIN_IMG, transform=tf_train)
    ds_valid = SphereDataset(image_dir=DIR_VALID_IMG, transform=tf_valid)

    # 3. Create data loaders
    loader_args = dict(batch_size=args.batch_size, num_workers=5, pin_memory=True)
    train_loader = DataLoader(ds_train, shuffle=True, **loader_args)
    val_loader = DataLoader(ds_valid, shuffle=False, drop_last=True, **loader_args)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=2)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    criterion = nn.BCEWithLogitsLoss()

    wandb.init(
        project="U-Net-tmp",
        config=dict(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            amp=args.amp,
        ),
    )

    logging.info(
        f"""Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.lr}
        Training size:   {len(ds_train)}
        Validation size: {len(ds_valid)}
        Device:          {device.type}
        Mixed Precision: {args.amp}
        """
    )

    try:
        for epoch in range(1, args.epochs + 1):
            with tqdm(total=len(ds_train), desc=f"{epoch}/{args.epochs}", unit="img") as pbar:

                # Training round
                for step, (images, true_masks) in enumerate(train_loader):
                    assert images.shape[1] == net.n_channels, (
                        f"Network has been defined with {net.n_channels} input channels, "
                        f"but loaded images have {images.shape[1]} channels. Please check that "
                        "the images are loaded correctly."
                    )

                    # transfer images to device
                    images = images.to(device=device)
                    true_masks = true_masks.unsqueeze(1).to(device=device)

                    # forward
                    with torch.cuda.amp.autocast(enabled=args.amp):
                        pred_masks = net(images)
                        train_loss = criterion(pred_masks, true_masks)

                    # backward
                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(train_loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    # update tqdm progress bar
                    pbar.update(images.shape[0])
                    pbar.set_postfix(**{"loss": train_loss.item()})

                    # log training metrics
                    wandb.log(
                        {
                            "train/epoch": epoch - 1 + step / len(train_loader),
                            "train/train_loss": train_loss,
                        }
                    )

                # Evaluation round
                val_score = evaluate(net, val_loader, device)
                scheduler.step(val_score)

                # log validation metrics
                wandb.log(
                    {
                        "val/val_score": val_score,
                    }
                )

            print(f"Train Loss: {train_loss:.3f}, Valid Score: {val_score:3f}")

            # save weights when epoch end
            Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(CHECKPOINT_DIR / "checkpoint_epoch{}.pth".format(epoch)))
            logging.info(f"Checkpoint {epoch} saved!")

    except KeyboardInterrupt:
        torch.save(net.state_dict(), "INTERRUPTED.pth")
        logging.info("Saved interrupt")
        raise


if __name__ == "__main__":
    main()
