import logging

import albumentations as A
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.utils.dataset import SphereDataset
from unet import UNet
from utils.dice import dice_coeff
from utils.paste import RandomPaste


def main():
    # setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # setup wandb
    wandb.init(
        project="U-Net",
        config=dict(
            DIR_TRAIN_IMG="/home/lilian/data_disk/lfainsin/val2017",
            DIR_VALID_IMG="/home/lilian/data_disk/lfainsin/smoltrain2017/",
            DIR_SPHERE_IMG="/home/lilian/data_disk/lfainsin/spheres/Images/",
            DIR_SPHERE_MASK="/home/lilian/data_disk/lfainsin/spheres/Masks/",
            FEATURES=[64, 128, 256, 512],
            N_CHANNELS=3,
            N_CLASSES=1,
            AMP=True,
            PIN_MEMORY=True,
            BENCHMARK=True,
            DEVICE="cuda",
            WORKERS=8,
            EPOCHS=5,
            BATCH_SIZE=16,
            LEARNING_RATE=1e-4,
            IMG_SIZE=512,
            SPHERES=5,
        ),
        settings=wandb.Settings(
            code_dir="./src/",
        ),
    )

    # create device
    device = torch.device(wandb.config.DEVICE)

    # enable cudnn benchmarking
    torch.backends.cudnn.benchmark = wandb.config.BENCHMARK

    # 0. Create network
    net = UNet(n_channels=wandb.config.N_CHANNELS, n_classes=wandb.config.N_CLASSES, features=wandb.config.FEATURES)
    wandb.config.PARAMETERS = sum(p.numel() for p in net.parameters() if p.requires_grad)

    # transfer network to device
    net.to(device=device)

    # 1. Create transforms
    tf_train = A.Compose(
        [
            A.Resize(wandb.config.IMG_SIZE, wandb.config.IMG_SIZE),
            A.Flip(),
            A.ColorJitter(),
            RandomPaste(wandb.config.SPHERES, wandb.config.DIR_SPHERE_IMG, wandb.config.DIR_SPHERE_MASK),
            A.GaussianBlur(),
            A.ISONoise(),
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ],
    )
    tf_valid = A.Compose(
        [
            A.Resize(wandb.config.IMG_SIZE, wandb.config.IMG_SIZE),
            RandomPaste(wandb.config.SPHERES, wandb.config.DIR_SPHERE_IMG, wandb.config.DIR_SPHERE_MASK),
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ],
    )

    # 2. Create datasets
    ds_train = SphereDataset(image_dir=wandb.config.DIR_TRAIN_IMG, transform=tf_train)
    ds_valid = SphereDataset(image_dir=wandb.config.DIR_VALID_IMG, transform=tf_valid)
    # ds_train_bg20k = SphereDataset(image_dir="/home/lilian/data_disk/lfainsin/BG-20k/train/", transform=tf_train)
    # ds_valid_bg20k = SphereDataset(image_dir="/home/lilian/data_disk/lfainsin/BG-20k/testval/", transform=tf_valid)

    # ds_train = torch.utils.data.ChainDataset([ds_train_coco, ds_train_bg20k])
    # ds_valid = torch.utils.data.ChainDataset([ds_valid_coco, ds_valid_bg20k]) # TODO: modifier la classe SphereDataset pour prendre plusieurs dossiers

    # 3. Create data loaders
    train_loader = DataLoader(
        ds_train,
        shuffle=True,
        batch_size=wandb.config.BATCH_SIZE,
        num_workers=wandb.config.WORKERS,
        pin_memory=wandb.config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        ds_valid,
        shuffle=False,
        drop_last=True,
        batch_size=wandb.config.BATCH_SIZE,
        num_workers=wandb.config.WORKERS,
        pin_memory=wandb.config.PIN_MEMORY,
    )

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for amp
    optimizer = torch.optim.RMSprop(net.parameters(), lr=wandb.config.LEARNING_RATE, weight_decay=1e-8, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=2)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=wandb.config.AMP)
    criterion = torch.nn.BCEWithLogitsLoss()

    # save model.pth
    torch.save(net.state_dict(), "checkpoints/model-0.pth")
    artifact = wandb.Artifact("pth", type="model")
    artifact.add_file("checkpoints/model-0.pth")
    wandb.run.log_artifact(artifact)

    # save model.onxx
    dummy_input = torch.randn(
        1, wandb.config.N_CHANNELS, wandb.config.IMG_SIZE, wandb.config.IMG_SIZE, requires_grad=True
    ).to(device)
    torch.onnx.export(net, dummy_input, "checkpoints/model-0.onnx")
    artifact = wandb.Artifact("onnx", type="model")
    artifact.add_file("checkpoints/model-0.onnx")
    wandb.run.log_artifact(artifact)

    # log gradients and weights four time per epoch
    wandb.watch(net, log_freq=(len(train_loader) + len(val_loader)) // 4)

    # print the config
    logging.info(f"wandb config:\n{yaml.dump(wandb.config.as_dict())}")

    try:
        for epoch in range(1, wandb.config.EPOCHS + 1):
            with tqdm(total=len(ds_train), desc=f"{epoch}/{wandb.config.EPOCHS}", unit="img") as pbar:

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
                    with torch.cuda.amp.autocast(enabled=wandb.config.AMP):
                        pred_masks = net(images)
                        train_loss = criterion(pred_masks, true_masks)

                    # backward
                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(train_loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    # compute metrics
                    pred_masks_bin = (torch.sigmoid(pred_masks) > 0.5).float()
                    accuracy = (true_masks == pred_masks_bin).float().mean()
                    dice = dice_coeff(pred_masks_bin, true_masks)
                    mae = torch.nn.functional.l1_loss(pred_masks_bin, true_masks)

                    # update tqdm progress bar
                    pbar.update(images.shape[0])
                    pbar.set_postfix(**{"loss": train_loss.item()})

                    # log metrics
                    wandb.log(
                        {
                            "epoch": epoch - 1 + step / len(train_loader),
                            "train/accuracy": accuracy,
                            "train/bce": train_loss,
                            "train/dice": dice,
                            "train/mae": mae,
                        }
                    )

                # Evaluation round
                net.eval()
                accuracy = 0
                val_loss = 0
                dice = 0
                mae = 0
                with tqdm(val_loader, total=len(ds_valid), desc="val", unit="img", leave=False) as pbar:
                    for images, masks_true in val_loader:

                        # transfer images to device
                        images = images.to(device=device)
                        masks_true = masks_true.unsqueeze(1).to(device=device)

                        # forward
                        with torch.inference_mode():
                            masks_pred = net(images)

                        # compute metrics
                        val_loss += criterion(pred_masks, true_masks)
                        mae += torch.nn.functional.l1_loss(pred_masks_bin, true_masks)
                        masks_pred_bin = (torch.sigmoid(masks_pred) > 0.5).float()
                        accuracy += (true_masks == pred_masks_bin).float().mean()
                        dice += dice_coeff(masks_pred_bin, masks_true)

                        # update progress bar
                        pbar.update(images.shape[0])

                accuracy /= len(val_loader)
                val_loss /= len(val_loader)
                dice /= len(val_loader)
                mae /= len(val_loader)

                # save the last validation batch to table
                table = wandb.Table(columns=["ID", "image", "ground truth", "prediction"])
                for i, (img, mask, pred) in enumerate(
                    zip(
                        images.to("cpu"),
                        masks_true.to("cpu"),
                        masks_pred.to("cpu"),
                    )
                ):
                    table.add_data(i, wandb.Image(img), wandb.Image(mask), wandb.Image(pred))

                # log validation metrics
                wandb.log(
                    {
                        "predictions": table,
                        "val/accuracy": accuracy,
                        "val/bce": val_loss,
                        "val/dice": dice,
                        "val/mae": mae,
                    },
                    commit=False,
                )

                # update hyperparameters
                net.train()
                scheduler.step(dice)

            # save weights when epoch end
            torch.save(net.state_dict(), f"checkpoints/model-{epoch}.pth")
            artifact = wandb.Artifact("pth", type="model")
            artifact.add_file(f"checkpoints/model-{epoch}.pth")
            wandb.run.log_artifact(artifact)

            # export model to onnx format
            dummy_input = torch.randn(1, 3, 512, 512, requires_grad=True).to(device)
            torch.onnx.export(net, dummy_input, f"checkpoints/model-{epoch}.onnx")
            artifact = wandb.Artifact("onnx", type="model")
            artifact.add_file(f"checkpoints/model-{epoch}.onnx")
            wandb.run.log_artifact(artifact)

        # stop wandb
        wandb.run.finish()

    except KeyboardInterrupt:
        torch.save(net.state_dict(), "INTERRUPTED.pth")
        raise


if __name__ == "__main__":
    main()  # TODO: fix toutes les metrics, loss, accuracy, dice...
