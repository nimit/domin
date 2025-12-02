IsaacSim pip install + IsaacLab

Note: This guide assumes [uv](https://docs.astral.sh/uv/getting-started/installation/) and [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) are installed. If not, replace `uv pip` commands with `pip`.
In any case, make sure to activate the respective virtual environments.

## Setting up the sim Environment

#### Installing isaac-sim v5.1 via pip

```sh
uv venv --python 3.11
source .venv/bin/activate
uv pip install --upgrade pip
uv pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
uv pip install datasets jsonlines av decord deepdiff # deepdiff required for dataset resume feature
```

Verify the installation by running

```sh
isaacsim isaacsim.exp.full.kit
```

#### Installing IsaacLab

Pre-requisites

```sh
sudo apt install cmake build-essential
```

Use `isaaclab.sh` script instead of `isaaclab-uv.sh` in absence of `uv`.

```sh
. third-party/IsaacLab/isaaclab.sh --install
```

Verify the installation by running

```sh
python third-party/IsaacLab/scripts/tutorials/00_sim/create_empty.py
```

#### For procedural generation

```sh
uv pip install scene-synthesizer[recommend] xatlas
```
