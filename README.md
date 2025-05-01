# diploma-thesis

Continuous Machine Learning, Concept Drift, MLOps, and Cooperative, Connected and Automated Mobility applications for my Diploma Thesis at ECE NTUA

## Environment Setup
This repository contains a `thesis.yaml` file that defines the conda environment for the project. The environment includes all the necessary dependencies and packages required to run the code.

[Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) or [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install) is required to create the environment.

### Installation

```bash
conda env create -f thesis.yaml
```
### Activation

```bash
conda activate thesis
```

### Deactivation

```bash
conda deactivate
```

## SUMO Setup

### Installation
SUMO is already installed on the conda environment.

### Environment Variables
To use SUMO, you need to set the environment variable `SUMO_HOME` to the path where SUMO is installed. It is also useful to add the SUMO binaries and tools to the `PATH` environment variable. You can do this by utilizing the `activate.d` scripts provided by conda that run every time you activate the environment.

```powershell
# For Windows
Set-Content -Path "$env:CONDA_PREFIX\etc\conda\activate.d\env_vars.ps1" -Value '$env:SUMO_HOME = "$env:CONDA_PREFIX\Lib\site-packages\sumo"
$env:PATH = "$env:PATH;$env:SUMO_HOME\bin;$env:SUMO_HOME\tools"'
```

```bash
# For Linux/MacOS
echo 'export SUMO_HOME="$CONDA_PREFIX/lib/python3.12/site-packages/sumo"' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export PATH="$PATH:$SUMO_HOME/bin:$SUMO_HOME/tools"' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

To restart the conda environment and reset the environment variables, run the following command:

```bash
conda deactivate
conda activate thesis
```

You can now check if the environment variables are set correctly.

```powershell
# For Windows
echo $env:SUMO_HOME
echo $env:PATH
```

```bash
# For Linux/MacOS
echo $SUMO_HOME
echo $PATH
```

### Tools
To take full advantage of all Python tools, you need to install the dependencies.

```powershell
# For Windows
pip install -r $env:SUMO_HOME/tools/requirements.txt
```

```bash
# For Linux/MacOS
pip install -r $SUMO_HOME/tools/requirements.txt
```
