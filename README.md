# diploma-thesis

Continuous Machine Learning, Concept Drift, MLOps, and Cooperative, Connected and Automated Mobility applications for my Diploma Thesis at ECE NTUA

## Environment Setup

All dependencies are declared in `pyproject.toml` and managed with [uv](https://docs.astral.sh/uv/).

### Install uv
```powershell
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# Linux/MacOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone and install dependencies
```bash
git clone https://github.com/geokyr/diploma-thesis
cd diploma-thesis
uv install
```

This will create a virtual environment `.venv` in the project root and install all the dependencies.

### Run your code
All scripts should be invoked via `uv run` so they use the right virtual environment.

```bash
uv run python scripts/simulation.py
```

Another option is to activate the virtual environment and run the scripts directly.

```powershell
# Windows
.venv\Scripts\activate
python scripts/simulation.py
```

```bash
# Linux/MacOS
source .venv/bin/activate
python scripts/simulation.py
```

To deactivate the virtual environment, run the following command.

```bash
deactivate
```
