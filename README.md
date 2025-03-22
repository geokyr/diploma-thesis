# diploma-thesis

Continuous Machine Learning, Concept Drift, MLOps, and Cooperative, Connected and Automated Mobility applications for my Diploma Thesis at ECE NTUA

## Environment Setup
This repository contains a `thesis.yaml` file that defines the conda environment for the project. The environment includes all the necessary dependencies and packages required to run the code. Miniconda or Anaconda is required to create the environment.

### Installation

```powershell
conda env create -f thesis.yaml
```
### Activation

```powershell
conda activate thesis
```

### Deactivation

```powershell
conda deactivate
```

## SUMO Setup

### Installation
SUMO is already installed on the conda environment.

### Environment Variables
To use SUMO, you need to set the environment variable `SUMO_HOME` to the path where SUMO is installed. It is also useful to add the SUMO binaries and tools to the `PATH` environment variable. You can do this by running the following command in your terminal:

```powershell
Set-Content -Path "$env:CONDA_PREFIX\etc\conda\activate.d\env_vars.ps1" -Value '$env:SUMO_HOME = "$env:CONDA_PREFIX\Lib\site-packages\sumo"
$env:PATH = "$env:PATH;$env:SUMO_HOME\bin;$env:SUMO_HOME\tools"'
```

This command creates a PowerShell script that correctly sets the `SUMO_HOME` and `PATH` environment variables every time you activate the conda environment. To restart the conda environment and reset the environment variables, run the following command:

```powershell
conda deactivate
conda activate thesis
```

You can check if the environment variables are set correctly by running the following command:

```powershell
echo $env:SUMO_HOME
echo $env:PATH
```

### Tools
To take full advantage of all Python tools, install the dependencies:

```powershell
pip install -r $env:SUMO_HOME/tools/requirements.txt
```
