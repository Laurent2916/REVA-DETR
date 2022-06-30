import logging

import albumentations as A
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from evaluate import evaluate
from src.utils.dataset import SphereDataset
from unet import UNet
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
            BENCHMARK=False,
            DEVICE="cuda",
            WORKERS=8,
            EPOCHS=5,
            BATCH_SIZE=16,
            LEARNING_RATE=1e-5,
            IMG_SIZE=512,
            SPHERES=5,
        ),
    )

    # create device
    device = torch.device(wandb.config.device)

    # enable cudnn benchmarking
    torch.backends.cudnn.benchmark = wandb.config.BENCHMARK

    # 0. Create network
    net = UNet(n_channels=wandb.config.N_CHANNELS, n_classes=wandb.config.N_CLASSES, features=wandb.config.FEATURES)
    wandb.config.parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)

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
    wandb.watch(net, log_freq=100)
    artifact = wandb.Artifact("pth", type="model")
    artifact.add_file("model.pth")
    wandb.run.log_artifact(artifact)

    # save model.onxx
    dummy_input = torch.randn(
        1, wandb.config.n_channels, wandb.config.IMG_SIZE, wandb.config.IMG_SIZE, requires_grad=True
    ).to(device)
    torch.onnx.export(net, dummy_input, "model.onnx")
    artifact = wandb.Artifact("onnx", type="model")
    artifact.add_file("model.onnx")
    wandb.run.log_artifact(artifact)

    # print the config
    logging.info(
        f"""wandb config:
        {yaml.dump(wandb.config.as_dict())}
        """
    )

    try:
        for epoch in range(1, wandb.config.EPOCHS + 1):
            with tqdm(total=len(ds_train), desc=f"{epoch}/{wandb.config.EPOCHS}", unit="img") as pbar:

                # Training round
                for step, (images, true_masks) in enumerate(train_loader):
                    assert images.shape[1] == net.N_CHANNELS, (
                        f"Network has been defined with {net.N_CHANNELS} input channels, "
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

            # save weights when epoch end
            torch.save(net.state_dict(), "model.pth")
            artifact = wandb.Artifact("pth", type="model")
            artifact.add_file("model.pth")
            wandb.run.log_artifact(artifact)

            # export model to onnx format
            dummy_input = torch.randn(1, 3, 512, 512, requires_grad=True).to(device)
            torch.onnx.export(net, dummy_input, "model.onnx")
            artifact = wandb.Artifact("onnx", type="model")
            artifact.add_file("model.onnx")
            wandb.run.log_artifact(artifact)

        wandb.run.finish()

    except KeyboardInterrupt:
        torch.save(net.state_dict(), "INTERRUPTED.pth")
        raise


if __name__ == "__main__":
    main()  # TODO: fix toutes les metrics, loss, accuracy, dice...
