<div align="right">

English | [中文](docs/README_ZH-CN.md)

</div>

# EAC-Net

![Model Structure](docs/imgs/model.png)

An open-source code for predicting atomic contributions charge density using Equivariant Message Passing Networks.

## Contents
- [EAC-Net](#eac-net)
  - [Contents](#contents)
  - [Features](#features)
  - [Installation](#installation)
  - [Quickstart](#quickstart)
    - [Pre-install (development)](#pre-install-development)
    - [Post-install (CLI)](#post-install-cli)
    - [Distributed Training with torchrun](#distributed-training-with-torchrun)
  - [Modes \& Arguments](#modes--arguments)
  - [Datasets](#datasets)
  - [Usage Examples](#usage-examples)
    - [Training the Model](#training-the-model)
    - [Testing the Model](#testing-the-model)
    - [Making Predictions](#making-predictions)
  - [EAC-mp](#eac-mp)
  - [Citation](#citation)
  - [License](#license)

## Features

- Higher accuracy and faster training and inference speed

- Outputting the charge density distribution of a single atom

- Support caching atomic local environment during inference phase to accelerate inference speed

- Support CHGCAR/H5 dataset format

- Support multi GPUs parallel training

## Installation

Since `EAC-Net` relies on libraries such as `torch_cluster`, to correctly install the `cuda` version for GPU training support, we recommend using `poetry` for installation.

```bash
pip install poetry
```

Alternatively, before running `pip install .`, please ensure that the `CUDA` versions of `torch`, `torch_cluster`, and `torch_scatter` libraries are installed. `torch_cluster` and `torch_scatter` can be found in their corresponding CUDA versions at [pytorch geometric](https://pytorch-geometric.com/whl).


```bash
# Clone the repository
git clone https://github.com/qin2xue3jian4/EAC-Net
cd EAC-Net

# Install the package with poetry and activate Environment
poetry install
eval $(poetry env activate)

# Or run the installation command after installing libraries such as torch
pip install .
```

> After installation, the `eac` entry point becomes available system-wide.

---

## Quickstart

### Pre-install (development)

```bash
# Launch directly via Python
python eac/entry.py <mode> [ARGUMENTS]
```

### Post-install (CLI)

```bash
# Use the installed CLI
eac <mode> [ARGUMENTS]
```

### Distributed Training with torchrun

```bash
# Using python script
torchrun --nproc_per_node 2 eac/entry.py train [ARGUMENTS]

# Using module invocation
torchrun --nproc_per_node 2 -m eac.entry train [ARGUMENTS]
```
> Currently, `torchrun` is only applicable in training mode.
---

## Modes & Arguments

EAC-Net supports three modes, each exposing its own set of arguments:

```bash
# List arguments and help
eac train -h
```

> **Note**: Replace `train` with `test` or `predict` to switch modes.

---

## Datasets

Although `EAC` supports the direct use of `CHGCAR` as a dataset, for efficiency, we recommend converting the data into an `h5` format dataset before training.

For conversion commands, please refer to:
```bash
python scripts/convert_dir.py <source_dir> <target_dir>
```
In addition, considering the particularity of the charge density dataset, `EAC` also supports sample datasets, which can be constructed by adding parameters such as `--random 50000` to the above command.

For more parameters and methods for screening samples, please read the `scripts/convert_dir.py` directly.

## Usage Examples

The following examples demonstrate how to use `EAC-Net` for charge density modeling of water molecular systems.

### Training the Model
```bash
eac train examples/water/input.yaml --out outputs/train --plot
```

During training, the program will automatically:
- Read training data specified in the input.yaml configuration file, automatically combining `data.root_dir` and `data.train.paths`, supporting wildcard patterns for convenient file path setup

- Save training logs and model files in the `outputs/train` directory

- Display training progress and loss function changes

To stop training prematurely, create a file named stop in the output directory, and the program will automatically terminate the training process.

### Testing the Model

After training is completed, use the following command to evaluate the model's performance on the validation set:
```bash
eac test --model outputs/train/models/model.pt --paths examples/water/data/8.h5 --paths examples/water/data/8.h5 --out outputs/test --plot
```

This command will:
- Load the trained model (`outputs/train/models/model.pt`)

- Test on the specified datasets (multiple datasets can be specified via the `--paths` parameter)

- Output test metrics (such as `MSE`, `MAE`, etc.)

- Generate diagonal comparison plots between predicted and actual results when the --plot parameter is set

### Making Predictions

Use the trained model to predict charge density for new structures:
```bash
eac predict --model outputs/train/models/model.pt --paths examples/water/POSCAR --out outputs/predict --num-workers 4 --ngfs 50*50*50 --probe-size 200
```

The prediction function supports:
- Input structure files in various formats including `POSCAR`, `h5`, `CHGCAR`, etc.

- Output charge density files in formats such as `CHGCAR`, `h5`, etc.

- Setting prediction grid size via the `--ngfs` parameter. For input files containing grid information (such as `h5`, `CHGCAR`), the program will automatically extract grid size from the structure files when `--ngfs` is not specified

- Configuring parallel prediction grid point quantity via the `--probe-size` parameter

## EAC-mp
We extended EAC-Net and trained a large pretrained model, EAC-mp, using CHGCAR files from the Materials Project.

- Dataset (processed / sampled): [EAC-Net Charge Density Dataset — Zenodo](https://zenodo.org/records/16990467)
- Pretrained weights: [EAC-mp-l5-3M.pt](https://store.aissquare.com/models/0a4060e2-f409-40ba-80c1-5a0af37f9230/eac-mp-l5-6000.pt)
- Training script: [input.yaml](examples/unicharge-mp/input.yaml)

## Citation
The preprint describing the `EAC-Net` software framework:
```
@misc{xuejian2025eacnetrealspacechargedensity,
      title={EAC-Net: Real-space charge density via equivariant atomic contributions}, 
      author={Qin Xuejian and Lv Taoyuze and Zhong Zhicheng},
      year={2025},
      eprint={2508.04052},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2508.04052}, 
}
```

## License

`EAC-Net` is licensed under the MIT License.
