"""Full assembly of the parts to form the complete network."""

import torch.nn as nn

from .blocks import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()

        # Network
        self.inc = DoubleConv(n_channels, features[0])

        self.downs = nn.ModuleList()
        for i in range(len(features) - 1):
            self.downs.append(
                Down(*features[i : i + 2]),
            )

        self.ups = nn.ModuleList()
        for i in range(len(features) - 1):
            self.ups.append(
                Up(*features[-1 - i : -3 - i : -1]),
            )

        self.outc = OutConv(features[0], n_classes)

    def forward(self, x):
        skips = []

        x = x.to(self.device)

        x = self.inc(x)

        for down in self.downs:
            skips.append(x)
            x = down(x)

        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        x = self.outc(x)

        return x
