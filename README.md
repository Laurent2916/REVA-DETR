# sphereDetect

sphereDetect is a simple neural network, based on a Mask R-CNN, to detect spherical landmarks for image calibration.

## Built with

- [Python](https://www.python.org/)

### Frameworks

- [PyTorch](https://pytorch.org/)
- [TorchVision](https://pytorch.org/vision/stable/index.html)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [PyTorch Metrics](https://torchmetrics.readthedocs.io/en/stable/)
- [PyTorch Lightning Bolts](https://www.pytorchlightning.ai/bolts)
- [ONNXRuntime](https://onnxruntime.ai/)

### Tools

- [Poetry](https://python-poetry.org/)
- [Docker](https://www.docker.com/)
- [VSCode](https://code.visualstudio.com/)
    - [ms-python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
    - [Python Docstring Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)
    - [Conventional Commits](https://marketplace.visualstudio.com/items?itemName=vivaxy.vscode-conventional-commits)
    - [Remote container](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
    - [EditorConfig](https://marketplace.visualstudio.com/items?itemName=EditorConfig.EditorConfig)
    - [Docker](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker)

## Getting started (with docker)

TODO

## Getting started (without docker)

### Installation

Clone the repository:
```bash
git clone git@git.inpt.fr:fainsil/pytorch-reva.git
cd pytorch-reva
```

Install the dependencies:
```bash
poetry install --with all
```

### Usage

Activate python environment:
```bash
poetry shell
```

Start and configure Weights & Biases local server:
```bash
wandb server start
wandb login
```

Start a training:
```bash
python src/train.py
```

## License

Distributed under the [MIT](https://choosealicense.com/licenses/mit/) license. \
See [`LICENSE`](https://git.inpt.fr/fainsil/pytorch-reva/-/blob/master/LICENSE) for more information.

## Contact

Laurent Fainsin _[fɛ̃zɛ̃]_ \
\<[laurent@fainsin.bzh](mailto:laurent@fainsin.bzh)\>
