# sphereDetect

sphereDetect is a simple neural network, based on a Mask R-CNN, to detect spherical landmarks for image calibration.

## Built with

- [Python](https://www.python.org/)

### Frameworks

- [PyTorch](https://pytorch.org/)
- [TorchVision](https://pytorch.org/vision/stable/index.html)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [PyTorch Lightning Bolts](https://www.pytorchlightning.ai/bolts)
- [PyTorch Metrics](https://torchmetrics.readthedocs.io/en/stable/)
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
    - [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

## Getting started (with docker and vscode)

### Requirements

- [Git](https://git-scm.com/)
- [Docker](https://www.docker.com/)
- [NVIDIA-Docker](https://github.com/NVIDIA/nvidia-docker)

### Installation

Clone the repository:
```bash
git clone git@git.inpt.fr:fainsil/pytorch-reva.git
```

Start VS Code:
```bash
vscode pytorch-reva
```

Install the [Remote Development extension pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack). \
Modify variables `UID` and `GID` in [`.devcontainer/Dockerfile`](https://git.inpt.fr/fainsil/pytorch-reva/-/blob/master/.devcontainer/Dockerfile#L6) if necessary.
Reopen the workspace in [devcontainer mode](https://code.visualstudio.com/docs/remote/containers).

### Usage

Configure [Weights & Biases (local) server](https://docs.wandb.ai/guides/self-hosted/local) at <http://localhost:8080>, and login:
```bash
wandb login --host http://localhost:8080
```

Press `F5` to launch `src/train.py` in debug mode (with breakpoints, slower) \
or press `Ctrl+F5` to launch `src/train.py` in release mode.

## Getting started (without docker)

### Requirements

- [Git](https://git-scm.com/)
- [Poetry](https://python-poetry.org/)
- [Python](https://www.python.org/)
- [Docker](https://www.docker.com/) (if local wandb server used)

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

Configure [Weights & Biases (local) server](https://docs.wandb.ai/guides/self-hosted/local), and login:
```bash
wandb server start
wandb login --host http://localhost:8080
```

Start a training:
```bash
python src/train.py
```

## License

Distributed under the [MIT](https://choosealicense.com/licenses/mit/) license. \
See [`LICENSE`](https://git.inpt.fr/fainsil/pytorch-reva/-/blob/master/LICENSE) for more information.

## Contact

Laurent Fainsin _[loʁɑ̃ fɛ̃zɛ̃]_ \
\<[laurent@fainsin.bzh](mailto:laurent@fainsin.bzh)\>
