# diploma-thesis

Continuous Machine Learning, Concept Drift, MLOps, and Cooperative, Connected and Automated Mobility applications for my Diploma Thesis at ECE NTUA

## Environment Setup
- Make sure you have conda (miniconda or anaconda) installed.
- Run the following command to create the environment:
```bash
conda env create -f thesis.yaml
```
- Activate the environment:
```bash
conda activate thesis
```
- Deactivate the environment when done:
```bash
conda deactivate
```

## SUMO Installation
For more information, refer to the relevant section on the [Official Docs](https://sumo.dlr.de/docs/Downloads.php).

To install SUMO, you can use the [64-bit installer for Windows](https://sumo.dlr.de/releases/1.22.0/sumo-win64-1.22.0.msi). Run it and follow the instructions. The default installation path is `C:\Program Files (x86)\Eclipse\Sumo\`. Make sure to check the option: Set SUMO_HOME and adapt PATH and PYTHONPATH.

To take full advantage of all Python tools install the dependencies:
```
pip install -r $env:SUMO_HOME/tools/requirements.txt
```
