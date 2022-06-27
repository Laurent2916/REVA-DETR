""" Full assembly of the parts to form the complete network """

from .blocks import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, features[0])

        self.downs = nn.ModuleList()
        for i in range(len(features) - 1):
            self.downs.append(
                Down(*features[i : i + 2]),
            )

        self.ups = nn.ModuleList()
        for i in range(len(features) - 1):
            self.ups.append(
                Up(*features[::-1][i : i + 2]),
            )

        self.outc = OutConv(features[0], n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
