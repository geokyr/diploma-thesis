# diploma-thesis

Continuous Machine Learning for Cooperative, Connected and Automated Mobility applications for my Diploma Thesis at ECE NTUA

## Overview
This work brings together:

- A **SUMO-based synthetic traffic dataset** for central Athens (train / test / rain scenarios)
- **Machine learning research** for Estimated Time of Arrival (ETA) prediction
- A full **drift-aware ML platform** with microservices, including a dashboard for monitoring, automated drift detection and adaptation, and support for multiple models

The **dataset** used in this work is publicly available on Zenodo, under [10.5281/zenodo.16950674](https://zenodo.org/records/16950674)

The repo is structured as a **monorepo**: dataset generation, ML experiments, platform code, and LaTeX sources for the thesis report and presentation all live here.

## Contributions
This diploma thesis includes contributions by **Georgios Angelis** and **Serafeim Tzelepis**.

The platform's conceptualization was a joint effort. The Fuel Consumption model was implemented by Georgios Angelis, while the Number of Stops model was implemented by Serafeim Tzelepis. Parts of the platform are also based on their work, including the Drift service (Serafeim Tzelepis) and the Summarizer service (Georgios Angelis).

## Quick Start
This project supports two main workflows:

1. **Dataset Generation & Machine Learning Research** - For running the dataset generation simulation and machine learning research experiments
2. **Platform** - For running the full Drift-Aware ML Platform

### Dataset Generation & Machine Learning Research
Use `uv` to manage dependencies and create isolated environments.

#### Install uv
```powershell
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# Linux/MacOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Clone and install dependencies
```bash
git clone https://github.com/geokyr/diploma-thesis
cd diploma-thesis
```

Install specific dependency groups based on your needs, for example:

```bash
# For dataset generation
uv sync --extra simulation

# For ML research experiments
uv sync --extra eta

# For all dependencies
uv sync --all-extras
```

#### Running Scripts
All scripts should be invoked via `uv run` to use the correct virtual environment.

```bash
# Run the dataset generation simulation
uv run simulation/simulation.py

# Run the baseline research experiment
uv run experiments/baseline_research.py
```

### Platform
Use Docker Compose to run the Drift-Aware ML Platform, which consists of the following services:
- **Backend** (port 8000) - Main orchestration and control service
- **Predictor** (ports 8001, 8002, 8003) - ETA, Fuel, and Stops prediction services
- **Drift** (port 8004) - Concept drift detection service
- **Summarizer** (port 8005) - LLM summarization service
- **Frontend** (port 8080) - Admin Dashboard and User Interface

#### Prerequisites
- Docker and Docker Compose installed
- API key for OpenRouter, used in the summarizer service (*free models are used, so no charges are incurred*)

#### Configuration
Create a `.env` file in the project root with your API key:

```bash
OPENROUTER_API_KEY=your_api_key_here
```

Docker Compose will automatically load environment variables from this file.

#### Production Mode
```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Stop all services
docker compose down
```

Access the platform running locally at `http://localhost:8080`.

#### Development Mode
For development with hot-reload (code changes reflected immediately):

```bash
# Start services with dev configuration
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# View logs
docker compose logs -f

# Stop all services
docker compose down
```

Development mode mounts your local `thesis/` directory into containers, allowing you to modify code without rebuilding images.

## Repository Structure
- `appdata/` - Data, models, logs and other artifacts for the platform
- `docs/` - Documentation for the project
- `experiments/` - Machine learning research experiments
- `outputs/` - Outputs from the experiments (models, metrics, logs)
- `presentation/` - Presentation in LaTeX
- `report/` - Report in LaTeX
- `simulation/` - SUMO configuration files and datasets
- `thesis/` - Python package including all the project code
  - `backend/` - Main orchestrator and control service
  - `common/` - Shared modules and configuration
  - `drift/` - Concept drift detection service
  - `eta/` - ETA prediction task
  - `frontend/` - Admin Dashboard and User Interface
  - `fuel/` - Fuel consumption prediction task
  - `predictor/` - Model prediction services
  - `simulation/` - Dataset generation simulation with SUMO
  - `stops/` - Number of stops prediction task
  - `summarizer/` - LLM summarization service
