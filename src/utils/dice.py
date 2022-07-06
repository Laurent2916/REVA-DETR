import torch


def dice_score(inputs, targets, smooth=1, logits=True):
    # comment out if your model contains a sigmoid or equivalent activation layer
    if logits:
        inputs = torch.sigmoid(inputs)

    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    return (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)


def dice_loss(inputs, targets, smooth=1, logits=True):
    return 1 - dice_score(inputs, targets, smooth, logits)
