import logging

import albumentations as A
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchmetrics import Dice
from tqdm import tqdm

import wandb
from src.utils.dataset import SphereDataset
from unet import UNet
from utils.dice import DiceLoss
from utils.paste import RandomPaste

class_labels = {
    1: "sphere",
}

if __name__ == "__main__":
    # setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # setup wandb
    wandb.init(
        project="U-Net",
        config=dict(
            DIR_TRAIN_IMG="/home/lilian/data_disk/lfainsin/train/",
            DIR_VALID_IMG="/home/lilian/data_disk/lfainsin/val/",
            DIR_TEST_IMG="/home/lilian/data_disk/lfainsin/test/",
            DIR_SPHERE_IMG="/home/lilian/data_disk/lfainsin/spheres/Images/",
            DIR_SPHERE_MASK="/home/lilian/data_disk/lfainsin/spheres/Masks/",
            FEATURES=[64, 128, 256, 512],
            N_CHANNELS=3,
            N_CLASSES=1,
            AMP=True,
            PIN_MEMORY=True,
            BENCHMARK=True,
            DEVICE="cuda",
            WORKERS=7,
            EPOCHS=1001,
            BATCH_SIZE=16,
            LEARNING_RATE=1e-4,
            WEIGHT_DECAY=1e-8,
            MOMENTUM=0.9,
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
    ds_test = SphereDataset(image_dir=wandb.config.DIR_TEST_IMG)

    # 2.5. Create subset, if uncommented
    # ds_train = torch.utils.data.Subset(ds_train, list(range(0, len(ds_train), len(ds_train) // 10000)))
    # ds_valid = torch.utils.data.Subset(ds_valid, list(range(0, len(ds_valid), len(ds_valid) // 100)))
    # ds_test = torch.utils.data.Subset(ds_test, list(range(0, len(ds_test), len(ds_test) // 100)))

    ds_train = torch.utils.data.Subset(ds_train, [0])
    ds_valid = torch.utils.data.Subset(ds_valid, [0])
    ds_test = torch.utils.data.Subset(ds_test, [0])

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
    test_loader = DataLoader(
        ds_test,
        shuffle=False,
        drop_last=False,
        batch_size=1,
        num_workers=wandb.config.WORKERS,
        pin_memory=wandb.config.PIN_MEMORY,
    )

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for amp
    optimizer = torch.optim.RMSprop(
        net.parameters(),
        lr=wandb.config.LEARNING_RATE,
        weight_decay=wandb.config.WEIGHT_DECAY,
        momentum=wandb.config.MOMENTUM,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=2)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=wandb.config.AMP)
    criterion = torch.nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    # save model.onxx
    dummy_input = torch.randn(
        1, wandb.config.N_CHANNELS, wandb.config.IMG_SIZE, wandb.config.IMG_SIZE, requires_grad=True
    ).to(device)
    torch.onnx.export(net, dummy_input, "checkpoints/model.onnx")
    artifact = wandb.Artifact("onnx", type="model")
    artifact.add_file("checkpoints/model-0.onnx")
    wandb.run.log_artifact(artifact)

    # log gradients and weights four time per epoch
    wandb.watch(net, log_freq=100)

    # print the config
    logging.info(f"wandb config:\n{yaml.dump(wandb.config.as_dict())}")

    # wandb init log
    wandb.log(
        {
            "train/learning_rate": optimizer.state_dict()["param_groups"][0]["lr"],
        },
        commit=False,
    )

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

                    # compute loss

                    # backward
                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(train_loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    # compute metrics
                    pred_masks_bin = (torch.sigmoid(pred_masks) > 0.5).float()
                    accuracy = (true_masks == pred_masks_bin).float().mean()
                    dice = dice_loss.coeff(pred_masks, true_masks)
                    mae = torch.nn.functional.l1_loss(pred_masks_bin, true_masks)

                    # update tqdm progress bar
                    pbar.update(images.shape[0])
                    pbar.set_postfix(**{"loss": train_loss.item()})

                    # log metrics
                    wandb.log(
                        {
                            "epoch": epoch - 1 + step / len(train_loader),
                            "train/accuracy": accuracy,
                            "train/loss": train_loss,
                            "train/dice": dice,
                            "train/mae": mae,
                        }
                    )

                    if step and (step % 100 == 0 or step == len(train_loader)):
                        # Evaluation round
                        net.eval()
                        accuracy = 0
                        val_loss = 0
                        dice = 0
                        mae = 0
                        with tqdm(val_loader, total=len(ds_valid), desc="val.", unit="img", leave=False) as pbar2:
                            for images, masks_true in val_loader:

                                # transfer images to device
                                images = images.to(device=device)
                                masks_true = masks_true.unsqueeze(1).to(device=device)

                                # forward
                                with torch.inference_mode():
                                    masks_pred = net(images)

                                # compute metrics
                                val_loss += criterion(masks_pred, masks_true)
                                dice += dice_loss.coeff(pred_masks, true_masks)
                                masks_pred_bin = (torch.sigmoid(masks_pred) > 0.5).float()
                                mae += torch.nn.functional.l1_loss(masks_pred_bin, masks_true)
                                accuracy += (masks_true == masks_pred_bin).float().mean()

                                # update progress bar
                                pbar2.update(images.shape[0])

                        accuracy /= len(val_loader)
                        val_loss /= len(val_loader)
                        dice /= len(val_loader)
                        mae /= len(val_loader)

                        # save the last validation batch to table
                        table = wandb.Table(columns=["ID", "image", "ground truth", "prediction"])
                        for i, (img, mask, pred, pred_bin) in enumerate(
                            zip(
                                images.cpu(),
                                masks_true.cpu(),
                                masks_pred.cpu(),
                                masks_pred_bin.cpu().squeeze(1).int().numpy(),
                            )
                        ):
                            table.add_data(
                                i,
                                wandb.Image(img),
                                wandb.Image(mask),
                                wandb.Image(
                                    pred,
                                    masks={
                                        "predictions": {
                                            "mask_data": pred_bin,
                                            "class_labels": class_labels,
                                        },
                                    },
                                ),
                            )

                        # log validation metrics
                        wandb.log(
                            {
                                "val/predictions": table,
                                "train/learning_rate": optimizer.state_dict()["param_groups"][0]["lr"],
                                "val/accuracy": accuracy,
                                "val/loss": val_loss,
                                "val/dice": dice,
                                "val/mae": mae,
                            },
                            commit=False,
                        )

                        # update hyperparameters
                        net.train()
                        scheduler.step(train_loss)

                        # export model to onnx format when validation ends
                        dummy_input = torch.randn(1, 3, 512, 512, requires_grad=True).to(device)
                        torch.onnx.export(net, dummy_input, f"checkpoints/model-{epoch}-{step}.onnx")
                        artifact = wandb.Artifact("onnx", type="model")
                        artifact.add_file(f"checkpoints/model-{epoch}-{step}.onnx")
                        wandb.run.log_artifact(artifact)

                        # testing round
                        net.eval()
                        accuracy = 0
                        val_loss = 0
                        dice = 0
                        mae = 0
                        with tqdm(test_loader, total=len(ds_test), desc="test", unit="img", leave=False) as pbar3:
                            for images, masks_true in test_loader:

                                # transfer images to device
                                images = images.to(device=device)
                                masks_true = masks_true.unsqueeze(1).to(device=device)

                                # forward
                                with torch.inference_mode():
                                    masks_pred = net(images)

                                # compute metrics
                                val_loss += criterion(masks_pred, masks_true)
                                dice += dice_loss.coeff(pred_masks, true_masks)
                                masks_pred_bin = (torch.sigmoid(masks_pred) > 0.5).float()
                                mae += torch.nn.functional.l1_loss(masks_pred_bin, masks_true)
                                accuracy += (masks_true == masks_pred_bin).float().mean()

                                # update progress bar
                                pbar3.update(images.shape[0])

                        accuracy /= len(test_loader)
                        val_loss /= len(test_loader)
                        dice /= len(test_loader)
                        mae /= len(test_loader)

                        # save the last validation batch to table
                        table = wandb.Table(columns=["ID", "image", "ground truth", "prediction"])
                        for i, (img, mask, pred, pred_bin) in enumerate(
                            zip(
                                images.cpu(),
                                masks_true.cpu(),
                                masks_pred.cpu(),
                                masks_pred_bin.cpu().squeeze(1).int().numpy(),
                            )
                        ):
                            table.add_data(
                                i,
                                wandb.Image(img),
                                wandb.Image(mask),
                                wandb.Image(
                                    pred,
                                    masks={
                                        "predictions": {
                                            "mask_data": pred_bin,
                                            "class_labels": class_labels,
                                        },
                                    },
                                ),
                            )

                        # log validation metrics
                        wandb.log(
                            {
                                "test/predictions": table,
                                "test/accuracy": accuracy,
                                "test/loss": val_loss,
                                "test/dice": dice,
                                "test/mae": mae,
                            },
                            commit=False,
                        )

        # stop wandb
        wandb.run.finish()

    except KeyboardInterrupt:
        torch.save(net.state_dict(), "INTERRUPTED.pth")
        raise

# sapin de noel
