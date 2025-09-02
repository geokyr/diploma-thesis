# thesis

Continuous Machine Learning, Concept Drift, MLOps, and Cooperative, Connected and Automated Mobility applications for my Thesis at ECE NTUA

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
git clone https://github.com/geokyr/thesis
cd thesis
uv sync --all-extras
```

This will create a virtual environment `.venv` in the project root and install all the dependencies.

### Run your code
All scripts should be invoked via `uv run` so they use the right virtual environment.

```bash
uv run simulation/simulation.py
```

## Project Structure
- `experiments/` - Machine learning experiments
- `presentation/` - Presentation files and figures
- `resources/` - Various documents and images about the project
- `simulation/` - Simulation related files including the sumo network files
- `thesis/` - Python code package for the project
  - `backend/` - FastAPI backend service
  - `common/` - Common modules and config
  - `eta/` - Machine learning task
  - `frontend/` - Dash frontend service
  - `predictor/` - Predictor services
  - `simulation/` - Simulation pipeline
