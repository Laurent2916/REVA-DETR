import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    @staticmethod
    def coeff(inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        return (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    def forward(self, inputs, targets, smooth=1):
        return 1 - self.coeff(inputs, targets, smooth)
