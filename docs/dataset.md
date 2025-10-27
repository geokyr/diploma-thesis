# Dataset

## Table of Contents
1. [Overview](#overview)
2. [SUMO Simulation Framework](#sumo-simulation-framework)
3. [Network Generation](#network-generation)
4. [Traffic Demand Generation](#traffic-demand-generation)
5. [Iterative Development Process](#iterative-development-process)
6. [Concept Drift Scenario](#concept-drift-scenario)
7. [Pipeline Implementation](#pipeline-implementation)
8. [Dataset Characteristics](#dataset-characteristics)
9. [Reproducibility](#reproducibility)

## Overview

This document describes the pipeline for generating synthetic traffic datasets used in this thesis for ETA prediction, fuel consumption estimation, and number of stops prediction under concept drift conditions. The datasets were created using SUMO (Simulation of Urban MObility) and are publicly available on Zenodo.

**Dataset DOI:** [10.5281/zenodo.16950674](https://zenodo.org/records/16950674)

### Dataset Summary

The following table summarizes the characteristics of all three generated datasets:

| Metric | Train | Test | Rain |
|--------|-------|------|------|
| **Purpose** | Model training | Model evaluation | Concept drift testing |
| **Network** | Base (friction=1.0) | Base (friction=1.0) | Rain (friction=0.4) |
| **Random Seed** | 13 | 2025 | 314159 |
| **Simulation Duration** | 36000 s (10 hours) | 36000 s (10 hours) | 36000 s (10 hours) |
| **Total Trips** | 53,978 | 56,212 | 55,366 |
| **Average Trip Duration** | 205.22 s | 211.58 s | 247.00 s |
| **Average Trip Distance** | 1675.81 m | 1676.36 m | 1684.88 m |
| **Average Speed** | 30.24 km/h | 29.81 km/h | 25.74 km/h |
| **Total FCD Records** | 11,130,801 | 11,948,843 | 13,728,897 |
| **CSV Size** | 770.1 MB | 826.8 MB | 946.9 MB |
| **Parquet Size** | 233.3 MB | 247.4 MB | 282.3 MB |

**Output Format:** Floating Car Data (FCD) with per-timestep vehicle telemetry at 1-second resolution:
- `timestep`: Simulation time (seconds)
- `id`: Vehicle identifier
- `x`, `y`: Position coordinates on the network (meters)
- `speed`: Instantaneous speed (m/s)
- `lane`: Current lane ID
- `odometer`: Cumulative distance (m)
- `fuel`: Fuel consumption rate (mg/s)
- `waiting`: Time spent waiting since last stop (s)

### Dataset Versions

The dataset underwent four iterations during development:

| Version | Scenarios | Outputs | Format | Notes |
|---------|-----------|---------|--------|-------|
| **v1** | Base, Lane Closure, Rain | FCD, Emission (separate files) | CSV (train/test split per scenario) | Early exploration with lane closures |
| **v2** | Base, Rain | FCD, Emission | CSV (train/test split) | Simplified to two scenarios |
| **v3** | Train (base), Test (base), Rain | FCD only | CSV (3 files) | Final scenario structure |
| **v4** | Train (base), Test (base), Rain | FCD only | CSV + Parquet (3 files each) | **Current/Published version** |

**Current Version (v4)** is available on Zenodo and includes both CSV and Parquet formats for efficient storage and fast columnar access. Earlier versions explored alternative drift mechanisms (lane closures) and included emission outputs, which were ultimately deemed unnecessary for the machine learning tasks.

## SUMO Simulation Framework

### Why SUMO?

[SUMO (Simulation of Urban MObility)](https://sumo.dlr.de) is an open-source, microscopic traffic simulation package developed by the German Aerospace Center (DLR). It was selected for:

1. **Microscopic Modeling:** Individual vehicle behavior with realistic car-following models (Krauss model)
2. **OSM Integration:** Direct import from OpenStreetMap with road types, lanes, and speed limits
3. **Rich Telemetry:** Per-second FCD output with position, speed, fuel, waiting time
4. **Reproducibility:** Deterministic simulation with seed control
5. **Active Development:** Well-documented, widely used in academic research

### Network Statistics

The simulation area covers central Athens with the following characteristics:

| Metric | Value |
|--------|-------|
| **North-West Boundary** | (37.974745936977456, 23.725252771719436) |
| **South-East Boundary** | (37.988290142332225, 23.752735758169127) |
| **Size** | 2.42 km x 1.48 km |
| **Area** | 3.58 km² |
| **Edges** | 1184 |
| **Junctions** | 689 |
| **Traffic Lights** | 106 |
| **Total Road Length** | 68.17 km |
| **Average Edge Length** | 57.57 m |

## Network Generation

### Tool Selection: osmGet and osmBuild

While SUMO provides `osmWebWizard.py`, a web-based GUI that wraps the network generation process, we opted to use the underlying command-line tools, `osmGet.py` and `osmBuild.py`, directly. This decision was motivated by:

1. **Programmatic Control:** CLI invocation from Python scripts for reproducibility
2. **Parameter Access:** Fine-grained control over netconvert and polyconvert options not exposed by the web UI
3. **Automation:** Scripted pipeline execution without manual interaction

### Network Generation Overview

The network generation consists of two main stages performed once at the beginning of the pipeline:

1. **OSM Data Extraction:** Downloads OpenStreetMap data for the Athens bounding box, including roads, junctions, traffic lights, and building polygons
2. **Network Building:** Converts raw OSM data into SUMO-compatible network format with proper road types, traffic light control, and junction geometry

Additionally, a third variant is created for the rain scenario by applying a global friction reduction to all lanes in the base network.

Detailed implementation of these steps is described in the [Pipeline Implementation](#pipeline-implementation) section.

### Vehicle Classes

The simulation includes only passenger cars, excluding other vehicle types such as buses, trucks (HDV/LDV), and motorcycles. According to data from the Hellenic Statistical Authority (ELSTAT), the Greek vehicle fleet composition shows that buses and trucks each represent approximately 3-5% of registered vehicles, while motorcycles account for a more substantial proportion (approximately 10-15% in urban areas).

Despite the relatively higher representation of motorcycles in Athens traffic, they were excluded from the simulation for the following reasons:

1. **Behavioral Heterogeneity:** Motorcycles exhibit significantly different driving behaviors compared to passenger cars, including lane-splitting, different acceleration profiles, and distinct gap-acceptance patterns. Modeling these behaviors would require separate vehicle type configurations and validation procedures.

2. **Insufficient Training Data:** Even with 10-15% representation, the absolute number of motorcycle trips would provide inadequate samples for machine learning models to reliably learn class-specific prediction patterns, particularly when combined with the feature space dimensionality.

3. **Class Imbalance:** The resulting class imbalance would necessitate specialized handling techniques (e.g., oversampling, class-weighted loss functions) that add methodological complexity while providing marginal predictive improvements for the minority class.

4. **Task Focus:** The primary objective of this research is concept drift detection and model retraining strategies, not multi-modal traffic prediction. Restricting the vehicle fleet to passenger cars allows for clearer isolation of drift effects without confounding factors introduced by heterogeneous vehicle behaviors.

By focusing on the dominant vehicle class (passenger cars), the dataset maintains methodological simplicity while capturing the primary traffic dynamics relevant to the prediction tasks and research objectives.

## Traffic Demand Generation

### Traffic Generation Periods

Traffic demand follows a realistic hourly pattern designed to mimic real-world Athens traffic patterns observed in sources such as the Athens Mobility Observatory and similar traffic monitoring platforms. The pattern captures characteristic morning and evening rush hours with a midday decline typical of urban traffic. The base generation periods (in seconds between vehicle departures) are:

| Hour           | Period (s) | Vehicles/hr |
|----------------|------------|-------------|
| 08:00-09:00    | 0.50       | ~7200       |
| 09:00-10:00    | 0.55       | ~6545       |
| 10:00-11:00    | 0.65       | ~5538       |
| 11:00-12:00    | 0.75       | ~4800       |
| 12:00-13:00    | 0.80       | ~4500       |
| 13:00-14:00    | 0.80       | ~4500       |
| 14:00-15:00    | 0.75       | ~4800       |
| 15:00-16:00    | 0.65       | ~5538       |
| 16:00-17:00    | 0.65       | ~5538       |
| 17:00-18:00    | 0.60       | ~6000       |

**Pattern Characteristics:**
- **Morning Peak (08:00-10:00):** Highest traffic volume simulating morning rush
- **Midday Decline (10:00-15:00):** Reduced traffic
- **Evening Peak (15:00-18:00):** Increased traffic simulating evening rush

### Stochastic Traffic Volume

To introduce natural variability between simulations and avoid identical traffic patterns, each scenario applies Gaussian noise to the base traffic generation periods. Each base period is multiplied by a random value drawn from a normal distribution centered at 1.0 with standard deviation 0.01. This introduces some **stochastic variability** in traffic volumes while preserving the overall hourly pattern shape.

### Random Seeds

Each scenario uses a distinct random seed to ensure reproducibility while generating different traffic patterns:

| Scenario | Seed | Purpose |
|----------|------|---------|
| **train** | 13 | Training dataset with unique traffic patterns |
| **test** | 2025 | Evaluation dataset with different but stable patterns |
| **rain** | 314159 | Drift dataset with different patterns + rain network |

The seed controls:
1. Gaussian noise applied to traffic generation periods
2. Random trip generation (origin-destination pairs)
3. Departure and arrival positions on edges
4. Vehicle behavior stochasticity in SUMO

### Trip Generation

Random origin-destination pairs are generated for each scenario using SUMO's `randomTrips.py` tool. Trips are distributed according to the hourly traffic generation periods (after Gaussian noise application), validated for route feasibility, and assigned random departure/arrival positions on edges. This process is performed separately for each scenario (train, test, rain) using the respective random seeds.

Detailed implementation is described in the [Pipeline Implementation](#pipeline-implementation) section.

## Iterative Development Process

The simulation pipeline was developed iteratively to address challenges in creating realistic, diverse, and concept-drift-compatible datasets. Several drift mechanisms were explored before settling on the final approach.

### Initial Attempts: Alternative Drift Mechanisms

#### 1. Lane Closure Approach

This approach used SUMO's `closingLaneReroute` mechanism to mark specific lanes as closed, with vehicles calculating routes at insertion time. The critical issue: vehicles would stop at green traffic lights when approaching closed lanes, remaining stationary until SUMO's automatic teleportation mechanism activated after 300 seconds—a clear simulation failure.

The root cause was the absence of rerouting devices. Adding them would allow dynamic recalculation around closures, but this also meant that all vehicles would reroute based on real-time conditions, fundamentally altering traffic behavior compared to the base scenario. As a result, isolating the effect of closures from dynamic rerouting became impossible.

**Outcome:** Abandoned due to vehicle teleportation artifacts and inability to maintain comparable traffic behavior.

#### 2. Network Topology Modification

This approach used SUMO's `netedit` tool to physically remove closed lanes or edges, with junctions automatically recalculated. Removing edges reduced network capacity, causing vehicles to be inserted at different times due to congestion delays—breaking temporal comparability between scenarios.

The sensitivity was highly non-linear and unpredictable: closing major roads like Panepistimiou (a main thoroughfare) sometimes produced minimal impact, while closing minor edges would completely bottleneck the network. Network analysis metrics (betweenness centrality, edge importance) were used to identify strategic closures, but finding combinations that were realistic (e.g., metro construction, roadworks) while producing detectable but not catastrophic drift proved extremely difficult.

**Outcome:** Rejected due to timing shifts, non-linear sensitivity, and difficulty calibrating realistic yet effective closure scenarios.

#### 3. Vehicle Behavior Modification

This approach altered Krauss car-following model parameters (acceleration, deceleration, sigma), vehicle type distributions, driver imperfection, reaction times, and speed factors to simulate aggressive or cautious driving patterns. Parameters explored included modifying default vehicle type attributes such as `accel`, `decel`, `speedFactor`, `sigma`, and `tau`.

Behavior changes produced only subtle effects on aggregate traffic patterns, didn't correspond to clear real-world events (unlike rain or construction), and lacked ground truth for validation of "realistic" parameter ranges.

**Outcome:** Insufficient drift magnitude and poor interpretability.

### Key Insights

The iterative exploration led to important insights that motivated the final approach:
1. Drift mechanisms must preserve network validity and route feasibility
2. Drift should correspond to clear, real-world phenomena for interpretability
3. Physics-based parameters (like friction) have measurable, predictable effects

## Concept Drift Scenario

Based on the lessons learned from alternative approaches, the final drift mechanism uses **friction-based rain simulation**.

### Rain Scenario Design

The rain scenario introduces concept drift by simulating adverse weather conditions through reduced road surface friction. This approach was chosen because:

1. **Physical Realism:** Rain directly reduces tire-road friction, a well-understood phenomenon
2. **Interpretability:** Clear causal relationship between friction and vehicle behavior
3. **Measurable Impact:** Reduced friction increases braking distances, decreases acceleration, and lowers average speeds
4. **Network Preservation:** No topology changes—all routes remain valid
5. **SUMO Support:** Native friction parameter in lane definitions

### Implementation

A modified network is created by parsing the base network XML and applying a global friction reduction to all lanes. The friction parameter is set to 0.4 (down from the default 1.0) uniformly across all lanes in the network. This modified network is saved separately and used for the rain scenario simulation.

**Friction Value Selection:** The chosen value of 0.4 technically falls within the snow/ice range according to road surface friction coefficients, rather than the wet road range (μ ≈ 0.5-0.8). This decision was intentional: the objective was to produce a significant and detectable concept drift for model evaluation, rather than to precisely simulate realistic rain conditions. The 0.4 coefficient ensures measurable performance degradation while maintaining simulation stability and avoiding extreme scenarios that would lead to complete traffic collapse.

### Friction Effects

SUMO's friction parameter affects vehicle dynamics in several ways:

1. **Maximum Acceleration:** Reduced by the square root of the friction coefficient
2. **Maximum Deceleration:** Reduced by the square root of the friction coefficient
3. **Cornering Speed:** Lower friction requires slower speeds on curves
4. **Emergency Braking:** Increased stopping distances

**Result:** Vehicles in the rain scenario experience slower acceleration from stops, earlier and gentler braking, longer trip completion times, and different congestion patterns.

### Concept Drift Characteristics

The rain scenario introduces measurable distributional shifts compared to the baseline test scenario:

| Metric               | Test (Baseline) | Rain (Drift) | Change      |
|----------------------|-----------------|--------------|-------------|
| **Average Speed**    | 29.84 km/h      | 25.76 km/h   | **-13.68%** |
| **Trip Duration**    | 211.58 s        | 247.00 s     | **+16.74%** |
| **Trip Distance**    | 1677.53 m       | 1685.18 m    | **+0.46%**  |
| **Waiting Time**     | 19.78 s         | 22.02 s      | **+11.33%** |
| **Fuel Consumption** | 216300.76 mg    | 218780.58 mg | **+1.15%**  |

The most significant impacts are on average speed (reduced by 13.68%) and trip duration (increased by 16.74%), demonstrating clear concept drift that challenges machine learning models trained on baseline conditions.

## Pipeline Implementation

The pipeline is implemented as a modular Python package and orchestrated by a main script. It follows a linear execution model: network generation steps are performed once, followed by per-scenario loops for traffic generation and simulation.

### Pipeline Overview

The complete pipeline consists of nine steps:

1. **OSM Data Extraction** → Download OpenStreetMap data for Athens bounding box
2. **Network Building** → Convert OSM data to SUMO network format
3. **Rain Network Creation** → Generate friction-modified network for drift scenario
4. **GUI Settings** → Write SUMO-GUI visualization settings
5. **Configuration Files** → Generate simulation configuration files per scenario
6. **Trip Generation** → Create random origin-destination pairs per scenario
7. **Simulation Execution** → Run SUMO simulation per scenario
8. **Format Conversion** → Convert CSV output to Parquet format
9. **Exploratory Analysis** → Generate statistics and plots

### Execution Flow

**Phase 1: Network Preparation (Once)**
1. Extract OSM data for central Athens using the specified bounding box
2. Build SUMO network from OSM data with traffic lights, junctions, and road geometry
3. Create rain network variant with reduced friction (0.4)
4. Write GUI settings file for visualization

**Phase 2: Scenario Simulation (For Each: Train, Test, Rain)**
1. Create simulation configuration file referencing the appropriate network (base or rain)
2. Generate random trips with scenario-specific seed and traffic patterns
3. Execute SUMO simulation with FCD output enabled
4. Convert FCD CSV output to compressed Parquet format
5. Perform exploratory data analysis and generate summary plots

### Pipeline Steps (Detailed)

#### Step 1: OSM Data Extraction

Downloads OpenStreetMap data for the Athens bounding box using SUMO's `osmGet.py` tool. The extraction includes all specified road types (motorways through service roads), building polygons for visualization, and compressed output in gzip format. The bounding box coordinates define a ~4.3 km² area in central Athens.

#### Step 2: Network Building

Converts raw OSM data into SUMO-compatible network format using `osmBuild.py`. This process:
- Applies typemaps to convert OSM road types to SUMO edge types
- Removes redundant geometry points while preserving road shape
- Detects and properly models roundabouts and highway ramps
- Merges nearby junctions to simplify network topology
- Infers traffic light positions and configures them for actuated control
- Generates detailed junction corner geometry
- Preserves original street names for visualization
- Creates both network file (edges, lanes, junctions) and polygon file (buildings, areas)

#### Step 3: Rain Network Creation

Parses the base network XML file and modifies all lane elements to include a friction attribute set to 0.4. This is accomplished through XML manipulation: reading the gzipped network file, finding all lane elements, setting their friction attribute, and writing the modified network to a new gzipped file. This creates an identical network topology with altered physics parameters.

#### Step 4: GUI Settings

Writes a simple XML configuration file for SUMO-GUI that sets the visualization scheme to "real world" and sets a delay value for simulation playback. This file is referenced by the simulation configuration but is not required for headless execution.

#### Step 5: Configuration Files

Generates SUMO configuration files (`.sumocfg`) for each scenario by invoking SUMO with save-configuration mode. Each configuration specifies:
- Network file path (base network for train/test, rain network for rain scenario)
- Route/trip file path (scenario-specific)
- Additional files (polygons for visualization)
- Traffic light parameters (actuated control with jam threshold of 30)
- Rerouting parameters (adaptation steps: 18, interval: 10)
- FCD output path and attributes (id, x, y, speed, lane, odometer, fuel, waiting)
- Error handling (ignore route errors, continue simulation)
- Logging preferences (verbose output, statistics, no per-step logging)

#### Step 6: Trip Generation

Generates random origin-destination pairs using SUMO's `randomTrips.py` tool for each scenario. The generation process:
- Uses the appropriate network file (base or rain) for route validation
- Applies scenario-specific random seed for reproducibility
- Creates trips distributed over the 10-hour simulation period (0-36000 seconds)
- Uses hourly traffic generation periods (after Gaussian noise application)
- Randomly positions vehicle departures and arrivals on edges
- Validates that all trips have feasible routes through the network
- Outputs trip definitions in SUMO XML format

#### Step 7: Simulation Execution

Runs the SUMO microscopic traffic simulation using the generated configuration file. The simulation:
- Loads the network (base or rain variant)
- Processes all trips according to their departure times
- Simulates vehicle movement using the Krauss car-following model
- Applies friction effects (when using rain network)
- Enables friction device for all vehicles (probability = 1.0)
- Handles dynamic rerouting based on traffic conditions
- Controls traffic lights using actuated logic
- Exports FCD telemetry at 1-second intervals to CSV

The simulation runs for 36000 seconds (10 hours) and processes between 54,000-56,000 trips depending on the scenario.

#### Step 8: Format Conversion

Converts the FCD CSV output to Apache Parquet format for efficient storage and faster access. The conversion:
- Loads the CSV file into a pandas DataFrame
- Writes the DataFrame to Parquet using PyArrow engine
- Applies Snappy compression for optimal balance of compression ratio and speed
- Reduces file size by approximately 70% (e.g., 826.8 MB → 247.4 MB for test scenario)

The Parquet format enables efficient columnar access patterns used in subsequent data preprocessing and machine learning pipelines.

#### Step 9: Exploratory Data Analysis

Performs initial analysis on the FCD data to validate simulation quality and generate summary statistics. The analysis:
- Loads and preprocesses the Parquet FCD data
- Aggregates telemetry by hour for temporal analysis
- Reports FCD statistics (record count, mean speed, coverage)
- Extracts individual trips from continuous FCD records
- Reports trip statistics (count, duration, distance distributions)
- Generates plots:
  - Average speed and traffic generation period overlay per hour
  - Trip distance histogram
  - Trip duration histogram

Results are logged to console and plots are saved to the plots directory.

## Reproducibility

The pipeline is designed for full reproducibility and extensibility with all configuration parameters centralized in a YAML configuration file.

### Key Design Principles

- **Modular Pipeline:** Each step is an independent function with well-defined inputs and outputs
- **Centralized Configuration:** All parameters stored in a YAML file with structured dataclass access
- **Deterministic Execution:** Random seeds control all stochastic elements (traffic noise, trip generation, vehicle behavior)
- **Comprehensive Logging:** All commands, outputs, and errors are logged with timestamps
- **Automatic Validation:** Path existence and command success are validated at each step

### Customization Points

The pipeline can be easily adapted to generate new datasets by modifying configuration parameters. Key customization points include:

- **Geographic Area:** Change the bounding box to simulate any OSM-covered region
- **Traffic Demand:** Modify hourly traffic patterns (can extend to full 24-hour patterns)
- **Traffic Volume Noise:** Adjust the mean and standard deviation of the Gaussian noise
- **Random Seeds:** Use different seeds to generate alternative traffic patterns
- **Simulation Duration:** Extend or shorten the simulation timespan
- **Drift Parameters:** Adjust network friction to vary rain severity
- **Network Processing:** Modify netconvert options to change junction detection and traffic light logic
- **Rerouting Behavior:** Adjust adaptation steps and intervals to change vehicle re-routing aggressiveness
- **Vehicle Classes:** Include buses, trucks, or other vehicle types

The modular design allows researchers to reproduce the exact datasets described in this thesis or generate new variants for different experimental conditions.
