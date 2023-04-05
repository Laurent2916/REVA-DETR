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
python src/main.py predict --ckpt_path <path_to_checkpoint>
```

Quick and dirty way to export to `.onnx`:
```python
>>> from src.module import DETR
>>> checkpoint = "<path_to_checkpoint>"
>>> model = DETR.load_from_checkpoint(checkpoint)
>>> model.net.save_pretrained("hugginface_checkpoint")
```
```bash
python -m transformers.onnx --model=hugginface_checkpoint onnx_export/
```

## License

Distributed under the [MIT](https://choosealicense.com/licenses/mit/) license. \
See [`LICENSE`](https://github.com/Laurent2916/REVA-DETR/blob/master/LICENSE) for more information.

## Contact

Laurent Fainsin _[loʁɑ̃ fɛ̃zɛ̃]_ \
\<[laurent@fainsin.bzh](mailto:laurent@fainsin.bzh)\>
