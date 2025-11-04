# diploma-thesis

Continuous Machine Learning for Cooperative, Connected and Automated Mobility applications for my Diploma Thesis at ECE NTUA

## Quick Start
This project supports two main workflows:

1. **Dataset Generation & Machine Learning Research** - For running the dataset generation simulation and machine learning research experiments
2. **Platform** - For running the full Drift-Aware ML Platform

## Dataset Generation & Machine Learning Research
Use `uv` to manage dependencies and create isolated environments.

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
```

Install specific dependency groups based on your needs:

```bash
# For dataset generation
uv sync --extra simulation

# For ML research experiments
uv sync --extra eta

# For all dependencies
uv sync --all-extras
```

### Running Scripts
All scripts should be invoked via `uv run` to use the correct virtual environment.

#### Dataset Generation Simulation
```bash
# Run the dataset generation simulation
uv run simulation/simulation.py
```

#### Machine Learning Research Experiments
```bash
# Run the baseline research experiment
uv run experiments/baseline_research.py
```

### Dataset
The experiments use a dataset of **Synthetic 10-hour Traffic Simulations for Central Athens** generated with SUMO. The dataset contains three trace files (train, test, rain) with Floating Car Data (FCD) including vehicle coordinates, speed, fuel, and waiting time.

**The dataset will automatically download when running experiments.** Alternatively, you can manually download it from the [Zenodo Dataset Repository](https://zenodo.org/records/16950674).

Available formats: CSV and Parquet (~3.3 GB total)

## Platform
Use Docker Compose to run the Drift-Aware ML Platform, which consists of the following services:
- **Backend** (port 8000) - Main orchestration and control service
- **Predictor Services** (ports 8001, 8002, 8003) - ETA, Fuel, and Stops prediction services
- **Drift Service** (port 8004) - Concept drift detection service
- **Summarizer Service** (port 8005) - LLM summarization service
- **Frontend** (port 8080) - Admin Dashboard and User Interface

### Prerequisites
- Docker and Docker Compose installed
- API key for OpenRouter, used in the summarizer service (*free models are used, so no charges are incurred*)

### Configuration
Create a `.env` file in the project root with your API key:

```bash
OPENROUTER_API_KEY=your_api_key_here
```

Docker Compose will automatically load environment variables from this file.

### Production Mode
```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Stop all services
docker compose down
```

Access the platform at `http://localhost:8080`

### Development Mode
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

## Project Structure
- `appdata/` - Application data, models, logs and other artifacts for the platform
- `docs/` - Documentation for the project
- `experiments/` - Machine learning research experiments
- `outputs/` - Outputs from the experiments (models, metrics, logs)
- `presentation/` - Presentation in LaTeX
- `report/` - Report in LaTeX
- `resources/` - Various resources and files about the project
- `simulation/` - SUMO configuration files and datasets
- `thesis/` - Main Python package for the project
  - `backend/` - Main orchestration and control service
  - `common/` - Shared modules and configuration
  - `drift/` - Concept drift detection service
  - `eta/` - ETA prediction task
  - `frontend/` - Admin Dashboard and User Interface
  - `fuel/` - Fuel consumption prediction task
  - `predictor/` - Model prediction services
  - `simulation/` - Dataset generation simulation with SUMO
  - `stops/` - Number of stops prediction task
  - `summarizer/` - LLM summarization service
