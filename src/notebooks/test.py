import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

preds = [
    dict(
        boxes=torch.tensor(
            [
                [880.0560, 41.7845, 966.9839, 131.3355],
                [1421.0029, 682.4420, 1512.7570, 765.2380],
                [132.0775, 818.5026, 216.0825, 1020.8573],
            ]
        ),
        scores=torch.tensor(
            [0.9989, 0.9936, 0.0932],
        ),
        labels=torch.tensor(
            [1, 1, 1],
        ),
    )
]
target = [
    dict(
        boxes=torch.tensor(
            [[879, 39, 1513, 766]],
        ),
        labels=torch.tensor(
            [1],
        ),
    )
]
metric = MeanAveragePrecision()
metric.update(preds, target)

from pprint import pprint

pprint(metric.compute())

# --------------------------------------------------------------------------

preds = [
    dict(
        boxes=torch.tensor(
            [
                [880.0560, 41.7845, 1500, 700.3355],
            ]
        ),
        scores=torch.tensor(
            [0.9989],
        ),
        labels=torch.tensor(
            [1],
        ),
    )
]
target = [
    dict(
        boxes=torch.tensor(
            [[879, 39, 1513, 766]],
        ),
        labels=torch.tensor(
            [1],
        ),
    )
]
metric = MeanAveragePrecision()
metric.update(preds, target)

from pprint import pprint

pprint(metric.compute())
