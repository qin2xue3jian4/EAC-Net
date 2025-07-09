# EAC-Net
An open-source code for predicting atomic contributions charge density using Equivariant Message Passing Networks.

## Features

## Installation

```bash
# Clone the repository
git clone https://github.com/qin2xue3jian4/RealGridE3Net
cd RealGridE3Net

# Install dependencies
pip install -r requirements.txt

# Install the package
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
eacnet <mode> [ARGUMENTS]
```

### Distributed Training with torchrun

```bash
# Using python script
torchrun --nproc_per_node 2 eac/entry.py train [ARGUMENTS]

# Using module invocation
torchrun --nproc_per_node 2 -m eac.entry train [ARGUMENTS]
```

---

## Modes & Arguments

EACNet supports three modes, each exposing its own set of arguments:

```bash
# List arguments and help
eac train -h
```

> **Note**: Replace `train` with `test` or `predict` to switch modes.

---

## License

MIT © Your Organization
