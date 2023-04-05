# Neural sphere detection in images for lighting calibration

# Installation

Clone the repository:
```bash
git clone https://github.com/Laurent2916/REVA-DETR.git
cd REVA-DETR/
```

Install and activate the environment:
```bash
micromamba install -f environment.yml
micromamba activate qcav
```

## Usage

Everything is managed thanks to [Lightning CLI](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.cli.LightningCLI.html#lightning.pytorch.cli.LightningCLI) !

Start a training:
```bash
python src/main.py fit
```

Start inference on images:
```bash
python src/main predict --ckpt_path <path_to_checkpoint>
```

## License

Distributed under the [MIT](https://choosealicense.com/licenses/mit/) license. \
See [`LICENSE`](https://github.com/Laurent2916/REVA-DETR/blob/master/LICENSE) for more information.

## Contact

Laurent Fainsin _[loʁɑ̃ fɛ̃zɛ̃]_ \
\<[laurent@fainsin.bzh](mailto:laurent@fainsin.bzh)\>
