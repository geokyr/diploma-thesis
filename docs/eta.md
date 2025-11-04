# ETA

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Experimental Methodology](#experimental-methodology)
- [Transformation Experiments](#transformation-experiments)
- [Feature Engineering](#feature-engineering)
- [Feature Selection](#feature-selection)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Final Model](#final-model)

## Overview
This document describes the complete research process for developing a machine learning model for Estimated Time of Arrival (ETA) prediction in urban environments.

**Research Objectives:**
- Develop an accurate ETA prediction model for urban trips
- Compare multiple ML approaches and feature engineering strategies
- Ensure model robustness through comprehensive evaluation

**Key Results:**
- Final Model: LightGBM
- Best MAE: 26.51 seconds
- Best MAPE: 12.77%
- Training Time: 2.95 seconds

## Dataset

### Data Source
This research utilizes synthetic traffic data generated through microscopic traffic simulation using SUMO (Simulation of Urban MObility) for a 3.58 km² area of central Athens. The simulation generates Floating Car Data (FCD) with per-second vehicle telemetry over a 10-hour period, producing approximately 11.1 million timestep records.

The training dataset comprises 53978 vehicle trips collected under standard traffic conditions on a base network (friction coefficient = 1.0). Traffic demand follows realistic hourly patterns derived from Athens Mobility Observatory data, with morning and evening peak periods reflecting typical urban congestion patterns.

For comprehensive details on the simulation pipeline, network characteristics, traffic demand modeling, and concept drift scenarios, refer to the [Dataset Generation Documentation](dataset.md).

### From FCD to Trip-Level Features
The raw FCD output consists of per-second telemetry records with the following schema:

| Field | Description | Unit |
|-------|-------------|------|
| `timestep` | Simulation time | seconds |
| `id` | Vehicle identifier | - |
| `x`, `y` | Network coordinates | meters |
| `speed` | Instantaneous speed | m/s |
| `lane` | Current lane ID | - |
| `odometer` | Cumulative distance | meters |
| `fuel` | Fuel consumption rate | mg/s |
| `waiting` | Waiting time since departure | seconds |

To construct a trip-level dataset suitable for machine learning, the FCD records were aggregated by vehicle identifier (`id`) and transformed through the following process:

**Aggregation Pipeline:**

- **Grouping:** Records grouped by unique vehicle ID, representing individual trips from insertion to removal
- **Temporal Features:** Extract first and last timestep per vehicle to derive:
  - `time_start`: Trip departure timestamp (seconds)
  - `duration`: Trip completion time (last timestep - first timestep)
- **Spatial Features:** Extract first and last position per vehicle to derive:
  - `source_x`, `source_y`: Origin coordinates (meters)
  - `destination_x`, `destination_y`: Destination coordinates (meters)
- **Distance Calculation:** Extract final odometer reading per vehicle:
  - `distance`: Total route distance traveled (meters)

**Filtering Criteria:**

To ensure data quality and model reliability, trips were filtered based on the following criteria:
- **Minimum Duration:** `duration ≥ 30 seconds` (exclude extremely short trips)
- **Minimum Distance:** `distance ≥ 200 meters` (exclude stationary or negligible movement)

These thresholds remove edge cases such as vehicles that failed to complete routes, experienced immediate insertion errors, or represented non-meaningful trips that would introduce noise into the training process.

**Resulting Dataset:**

The transformation yields a trip-level dataset with 53229 samples, where each row represents a complete vehicle trip characterized by:
- **Origin-Destination Pair:** Source and destination coordinates
- **Temporal Context:** Departure timestamp
- **Route Characteristics:** Distance traveled and trip duration (target variable)

This trip-level representation forms the foundation for feature engineering and model training, enabling prediction of trip duration based on spatial, temporal, and route characteristics.

## Experimental Methodology

### Experiment Tracking System
A systematic experiment tracking framework was implemented to manage the iterative research process and ensure reproducibility across all experiments.

**Architecture:**

The research pipeline consists of three main components:

- **Experiment Scripts** (`experiments/` directory)
  - Dedicated Python script for each experiment configuration
  - Minimal boilerplate code through shared library functions
  - Clear naming convention (e.g., `baseline_research.py`, `features_spatial.py`, `tuning_lightgbm.py`)

- **Reusable Library** (`thesis/` package)
  - Centralized feature engineering logic
  - Common model training and evaluation utilities
  - Dataset loading and preprocessing functions
  - Cross-validation and stratification helpers
  - Eliminates code duplication across experiments

- **Automated Output Management** (`outputs/` directory)
  - Each experiment automatically creates a structured output directory
  - Standard subdirectories: `models/`, `logs/`, `results/`
  - Trained model artifacts saved as joblib files for reproducibility
  - Performance metrics stored in JSON format for programmatic access
  - Execution logs capture full experiment configuration and runtime details

**Configuration Management:**

- Centralized configuration in a YAML file
- Shared settings: dataset paths, cross-validation parameters, random seeds
- Consistent experimental setup across all runs
- Easy modification of global parameters without touching experiment code

**Reproducibility Features:**

- Fixed random seeds for all stochastic operations
- Version-controlled experiment scripts and library code
- Complete artifact preservation (models, metrics, logs)
- Deterministic cross-validation splits through stratified sampling

**Results Aggregation:**

- Automated result collection scripts (`experiments/results_*.py`)
- Aggregates metrics across all experiments for comparison
- Generates comparative visualizations (MAE, MAPE, training time distributions)
- CSV exports for further analysis

This modular architecture enabled rapid experimentation while maintaining code quality, reproducibility, and systematic tracking of over 30 distinct experiment configurations throughout the research process.

### Cross-Validation Strategy
**Method:** Stratified 5-Fold Cross-Validation

**Stratification:** Based on trip duration (target variable)
- Ensures representative distribution across folds
- Critical for handling skewed duration distributions
- Prevents overfitting to specific duration ranges

**Data Usage:**
- Training set only used for model development
- Test and rain datasets reserved for final evaluation
- No data leakage between folds

### Evaluation Metrics
Primary metrics used across all experiments:

| Metric | Description | Why Used |
|--------|-------------|----------|
| **MAE** | Mean Absolute Error (seconds) | Direct interpretability, less sensitive to outliers |
| **MAPE** | Mean Absolute Percentage Error (\%) | Relative error, scale-independent |
| **Training Time** | Model fitting duration (seconds) | Practical deployment consideration |

### Baseline Models
Four model families were evaluated:

- **Linear Regression** - Simple baseline for comparison
- **CatBoost** - Gradient boosting with categorical feature handling
- **LightGBM** - Fast gradient boosting
- **XGBoost** - Robust gradient boosting

**Rationale:** Gradient boosting methods have demonstrated superior performance on tabular data with complex feature interactions, making them well-suited for regression tasks with spatial and temporal features. Linear Regression serves as a simple baseline to quantify the value of non-linear modeling approaches.

**Initial Baseline Results** (`baseline_research` with 6 original features):

| Model | MAE (s) | MAPE (%) | Training Time (s) |
|-------|---------|----------|-------------------|
| Linear Regression | 31.46 | 16.11 | 0.003 |
| LightGBM | 27.78 | 13.39 | 0.40 |
| XGBoost | 27.91 | 13.59 | 0.57 |
| CatBoost | 28.23 | 13.57 | 0.58 |

Linear Regression established a baseline MAE of 31.46s, which gradient boosting methods improved upon by 12-13% (3.2-3.7 seconds), confirming the value of non-linear modeling for this task. All subsequent experiments focused on the three gradient boosting models due to their superior performance.

## Transformation Experiments
Before conducting feature engineering, a series of transformation experiments were performed to determine whether transforming features or the target variable could improve model performance. Data transformations are commonly used to address skewness, normalize distributions, or improve model convergence.

### Tested Transformations

#### Feature Transformations
- **Log Transformation** (`transform_features_log`)
  - Applied `log(1 + x)` transformation to the `distance` feature
  - **Rationale:** Distance often exhibits right-skewed distribution; log transformation can normalize this
  - **Implementation:** Applied to training data, then to validation/test data

- **Standard Scaling** (`transform_features_standard`)
  - Applied standardization (zero mean, unit variance) to all features
  - **Rationale:** Ensures all features are on the same scale
  - **Implementation:** Fitted on training data, applied to validation/test data

#### Target Transformations
- **Log Transformation** (`transform_target_log`)
  - Applied `log(1 + y)` to travel time, with inverse `exp(y) - 1` for predictions
  - **Rationale:** Travel time may exhibit log-normal distribution

- **Box-Cox Transformation** (`transform_target_boxcox`)
  - Power transformation to make data more Gaussian-like
  - **Rationale:** Automatically finds optimal power parameter λ to normalize distribution

- **Quantile Transformation** (`transform_target_quantile`)
  - Maps data to uniform or normal distribution using quantiles
  - **Rationale:** Non-parametric transformation robust to outliers

### Experiment Results
All transformation experiments used the original 6 features (baseline configuration) with 5-fold stratified cross-validation:

| Experiment | Transformation | Average MAE (s) | Improvement | Best Model |
|------------|---------------|-------------|-------------|------------|
| `baseline_research` | None | 27.97 | — | LightGBM (27.78s) |
| `transform_features_log` | Log(distance) | 27.97 | 0.00s (0.00%) | LightGBM (27.78s) |
| `transform_features_standard` | Standardization | 27.98 | 0.01s (0.04%) | LightGBM (27.80s) |
| `transform_target_log` | Log(duration) | 27.92 | 0.05s (0.18%) | LightGBM (27.73s) |
| `transform_target_boxcox` | Box-Cox(duration) | 27.93 | 0.04s (0.14%) | LightGBM (27.72s) |
| `transform_target_quantile` | Quantile(duration) | 27.90 | 0.07s (0.25%) | LightGBM (27.71s) |

### Key Findings
- **Feature transformations ineffective:** Both log transformation and standard scaling of features showed no meaningful improvement over the baseline. This is expected for tree-based gradient boosting models, which are inherently invariant to monotonic transformations and feature scaling.
- **Target transformations marginal:** Target transformations showed slight improvements (0.04-0.07s), but the gains were minimal relative to baseline performance (0.14-0.25% improvement).
- **Added complexity not justified:** While target transformations provided small improvements, they introduce additional complexity in the prediction pipeline (requiring inverse transformation) and make model interpretation more difficult.
- **Tree-based model robustness:** Gradient boosting models handle the original feature and target distributions effectively without requiring explicit transformations, unlike linear models or neural networks that often benefit from normalization.

### Decision
**No transformations were applied in subsequent experiments.** The minimal performance gains did not justify the added complexity, and tree-based models demonstrated sufficient robustness to handle the untransformed data. This decision simplified the feature engineering pipeline and maintained model interpretability.

## Feature Engineering
Feature engineering was conducted systematically to evaluate the impact of different feature groups on prediction performance. Each feature group was tested independently by adding it to the original features, and all groups were also combined to assess their collective contribution.

### Feature Groups
The following feature groups were developed for ETA prediction:

#### Original Features - 6 features
Base features extracted directly from the simulation FCD data. These features serve as the baseline for all experiments.

- `source_x`, `source_y` - Trip origin coordinates (meters)
- `destination_x`, `destination_y` - Trip destination coordinates (meters)
- `time_start` - Trip start timestamp (seconds)
- `distance` - Route distance traveled (meters)

#### Temporal Features - 5 features
Time-based patterns extracted from trip start time. These features encode temporal patterns.

- `hour_bin` - Hour of day (0-10)
- `is_morning`, `is_noon`, `is_afternoon` - Time period indicators
- `is_rush_hour` - Peak traffic period indicator

#### Spatial Features - 17 features
Geometric and geographic relationships derived from coordinates and distances. These features capture urban structure and geometric relationships that influence travel time.

- `x_center`, `y_center` - Trip midpoint coordinates
- `x_difference`, `y_difference` - Coordinate deltas between origin and destination
- `euclidean_distance` - Straight-line distance between origin and destination
- `route_efficiency` - Ratio of euclidean distance to route distance
- `detour_length` - Difference between route distance and euclidean distance
- `trip_bearing`, `trip_bearing_sin`, `trip_bearing_cos` - Trip direction and trigonometric encoding
- `is_short_distance`, `is_medium_distance`, `is_long_distance` - Distance category indicators based on percentile thresholds
- `source_distance_from_city_center`, `destination_distance_from_city_center` - Radial distances from city center
- `trip_centrality_change` - Change in radial distance from origin to destination
- `trip_centrality` - Radial distance of trip midpoint from city center

#### Fourier Features - 16 features
Sinusoidal positional encoding of spatial coordinates. These features capture spatial patterns in the coordinate space.

- Sine and cosine transforms of coordinates at 2 frequency scales
- Applied to: `source_x`, `source_y`, `destination_x`, `destination_y`
- Pattern: `{coordinate}_sin_{0|1}` and `{coordinate}_cos_{0|1}`

#### Cell Features - 4 features
Spatial discretization into fixed-size grid cells. These features capture spatial patterns in the grid space.

- `source_cell_x`, `source_cell_y` - Origin grid cell indices
- `destination_cell_x`, `destination_cell_y` - Destination grid cell indices

**Parameters:** Cell size = 100 meters

#### Cluster Features - 2 features
K-Means clustering on coordinates to discretize spatial regions. These features capture spatial patterns in the cluster space.

- `source_cluster` - Cluster ID for trip origin (0-19)
- `destination_cluster` - Cluster ID for trip destination (0-19)

**Parameters:** K=20 clusters, fitted on training data coordinates

#### PCA Features - 4 features
Principal Component Analysis for dimensionality reduction. These features capture dominant spatial variance in a reduced dimensional space.

- `source_pca_1`, `source_pca_2` - First two principal components of origin coordinates
- `destination_pca_1`, `destination_pca_2` - First two principal components of destination coordinates

**Parameters:** 2 components, fitted on combined origin/destination coordinates

### Feature Engineering Experiments
Each feature group was evaluated through dedicated experiments to assess its contribution to prediction performance. The experiments followed a systematic approach:

- **Individual Group Testing:** Each feature group was added to the original features independently
- **Combined Testing:** All feature groups were combined to assess collective impact
- **Cross-Validation:** All experiments used 5-fold stratified cross-validation on the training dataset
- **Model Comparison:** Three gradient boosting models (CatBoost, LightGBM, XGBoost) were evaluated for each configuration

#### Experiment Results
The following table summarizes the average MAE (in seconds) across all models for each feature engineering experiment:

| Experiment | Feature Configuration | Total Features | Average MAE (s) | Improvement | Best Model |
|------------|----------------------|----------------|-------------|------------------------|------------|
| `baseline_research` | Original only | 6 | 27.97 | — | LightGBM (27.78s) |
| `features_temporal` | Original + Temporal | 11 | 27.84 | 0.13s (0.5%) | XGBoost (27.57s) |
| `features_spatial` | Original + Spatial | 21 | 27.77 | 0.20s (0.7%) | XGBoost (27.42s) |
| `features_fourier` | Original + Fourier | 22 | 27.69 | 0.28s (1.0%) | XGBoost (27.48s) |
| `features_cell` | Original + Cell | 10 | 27.76 | 0.21s (0.8%) | XGBoost (27.39s) |
| `features_cluster` | Original + Cluster | 8 | 27.87 | 0.10s (0.4%) | XGBoost (27.51s) |
| `features_pca` | Original + PCA | 10 | 27.73 | 0.24s (0.9%) | XGBoost (27.38s) |
| `features_all` | All features | 52 | 27.63 | 0.34s (1.2%) | XGBoost (27.33s) |

**Key Findings:**

- **All feature groups provide improvement:** Every feature group showed positive impact over the baseline, ranging from 0.10s to 0.34s MAE reduction.
- **Fourier features most effective individually:** The `features_fourier` experiment achieved the best single-group performance (27.69s MAE), demonstrating the value of positional encoding.
- **Combined features perform best:** The `features_all` experiment achieved the lowest MAE (27.63s), indicating complementary information across feature groups.
- **Model consistency:** XGBoost performed best on most feature configurations, though LightGBM was competitive and significantly faster to train.
- **Diminishing returns:** The improvement from combining all features (0.34s) is less than the sum of individual improvements, suggesting some overlap in captured information.

## Feature Selection
After evaluating the combined feature set (`features_all` with 52 features), systematic feature selection was performed to identify the most valuable features while reducing dimensionality, training time, and model complexity.

### Selection Methods
Four complementary feature importance methods were applied to rank all 52 features:

- **Gain-Based Importance**
  - Used LightGBM's built-in split gain importance
  - Measures feature contribution to loss reduction across all tree splits
  - Reflects how frequently and effectively each feature is used in decision trees

- **Permutation Importance**
  - Model-agnostic approach measuring feature impact
  - Computed increase in MAE when each feature is randomly shuffled
  - Captures actual predictive impact independent of model architecture

- **SHAP Values**
  - Shapley Additive Explanations from game theory
  - Quantifies each feature's contribution to individual predictions
  - Captures feature interactions and non-linear effects

- **Correlation Analysis**
  - Computed Pearson correlation between all feature pairs
  - Identified highly correlated features (|r| > 0.95)
  - Found 44 correlated pairs requiring resolution

### Combined Ranking
The three importance-based methods (gain, permutation, SHAP) were aggregated into a combined ranking:

- **Normalization:** Scores from each method normalized to [0, 1] range
- **Aggregation:** Mean rank computed across the three methods for each feature
- **Final Score:** Weighted combination of normalized importance and rank

**Top 20 Features by Combined Score:**

| Rank | Feature | Score | Category |
|------|---------|-------|----------|
| 1 | `distance` | 0.938 | Original |
| 2 | `is_short_distance` | 0.567 | Spatial |
| 3 | `euclidean_distance` | 0.520 | Spatial |
| 4 | `x_center` | 0.497 | Spatial |
| 5 | `is_long_distance` | 0.476 | Spatial |
| 6 | `detour_length` | 0.473 | Spatial |
| 7 | `time_start` | 0.428 | Original |
| 8 | `y_center` | 0.424 | Spatial |
| 9 | `destination_distance_from_city_center` | 0.412 | Spatial |
| 10 | `source_distance_from_city_center` | 0.396 | Spatial |
| 11 | `is_medium_distance` | 0.395 | Spatial |
| 12 | `source_cluster` | 0.384 | Cluster |
| 13 | `is_noon` | 0.378 | Temporal |
| 14 | `trip_centrality` | 0.368 | Spatial |
| 15 | `destination_pca_1` | 0.367 | PCA |
| 16 | `source_pca_2` | 0.359 | PCA |
| 17 | `destination_y_sin_1` | 0.358 | Fourier |
| 18 | `source_pca_1` | 0.335 | PCA |
| 19 | `destination_pca_2` | 0.308 | PCA |
| 20 | `destination_y` | 0.289 | Original |

**Key Observations:**
- Distance-based features (original + spatial) dominate the top 10 positions
- All 4 PCA features appear in the top 20, confirming their value for dimensionality reduction
- Cluster features rank moderately high (12th), capturing spatial zone patterns
- Original coordinate features rank lower (20+), but are retained due to PCA high correlations

### Selection Decision
Based on the combined ranking, correlation analysis, and domain knowledge, the following features were eliminated:

**Dropped Features (30 total):**

- **All Temporal Features (5):** Low importance scores; minimal impact in uniform synthetic traffic
  - Dropped: `hour_bin`, `is_morning`, `is_noon`, `is_afternoon`, `is_rush_hour`

- **All Fourier Features (16):** Low importance score; high correlation with coordinate features (r > 0.98); redundant positional encoding
  - Dropped: All sine/cosine transforms of `source_x`, `source_y`, `destination_x`, `destination_y`

- **All Cell Features (4):** Extremely high correlation with coordinates (r > 0.99); redundant with PCA features
  - Dropped: `source_cell_x`, `source_cell_y`, `destination_cell_x`, `destination_cell_y`

- **Redundant Spatial Features (7):** Low importance scores or high correlation with retained features
  - Dropped: `x_difference`, `y_difference`, `route_efficiency`, `trip_bearing`, `trip_bearing_sin`, `trip_bearing_cos`, `trip_centrality_change`

**Retained Features (22 total):**

- **Original (6):** `source_x`, `source_y`, `destination_x`, `destination_y`, `time_start`, `distance`
- **Spatial (10):** `x_center`, `y_center`, `euclidean_distance`, `detour_length`, `is_short_distance`, `is_medium_distance`, `is_long_distance`, `source_distance_from_city_center`, `destination_distance_from_city_center`, `trip_centrality`
- **Cluster (2):** `source_cluster`, `destination_cluster`
- **PCA (4):** `source_pca_1`, `source_pca_2`, `destination_pca_1`, `destination_pca_2`

### Performance Impact
The feature selection was validated through a dedicated experiment (`features_selection`) comparing against the full feature set:

| Configuration | Features | Average MAE (s) | Average MAPE (%) | Average Training Time (s) | Best Model |
|--------------|----------|-------------|----------|-------------------|------------|
| `features_all` | 52 | 27.63 | 13.26 | 0.81 | XGBoost (27.33s) |
| `features_selection` | 22 | 27.66 | 13.27 | 0.49 | XGBoost (27.33s) |
| **Change** | **-58%** | **+0.03s** | **+0.01%** | **-40%** | **Identical** |

**Key Results:**

- **Negligible accuracy loss:** Only 0.03s (0.1%) MAE increase despite removing 58% of features
- **Significant training speedup:** 40% reduction in training time (0.81s → 0.49s)
- **Maintained best performance:** XGBoost achieved identical MAE (27.33s) in both configurations
- **Improved efficiency:** 22 features provide 99.9% of the predictive power of 52 features
- **Enhanced interpretability:** Smaller, more focused feature set easier to understand and maintain

The feature selection successfully identified and retained the most informative features while eliminating redundancy and reducing computational cost.

## Hyperparameter Tuning
Systematic hyperparameter optimization was performed for all three gradient boosting models using Optuna with the Tree-structured Parzen Estimator (TPE) sampler. The optimization objective was to minimize Mean Absolute Error (MAE) across 5-fold stratified cross-validation.

### Tuning Strategy
A two-phase approach was employed to balance exploration and exploitation:

**Phase 1: Broad Search (100 trials per model)**
- Wide parameter ranges covering commonly effective values
- Parameter ranges informed by dataset characteristics (size, feature count)
- Goal: Identify promising regions of the hyperparameter space
- Experiments: `tuning_catboost`, `tuning_lightgbm`, `tuning_xgboost`

**Phase 2: Focused Search (50 trials per model)**
- Narrowed parameter ranges centered around Phase 1 best results
- Refined search in high-performing regions
- Goal: Fine-tune optimal configurations
- Experiments: `tuning_catboost_focused`, `tuning_lightgbm_focused`, `tuning_xgboost_focused`

The optimization process totaled 450 trials (150 per model) without early stopping, leveraging the fast training times enabled by the compact dataset

### Tuning Results
**Best Results by Model and Phase:**

| Model | Phase | Best MAE (s) | Best MAPE (%) | Training Time (s) | Trial # |
|-------|-------|-------------|---------------|-------------------|---------|
| CatBoost | Broad (100 trials) | 26.72 | 12.80 | 11.56 | 91 |
| CatBoost | Focused (50 trials) | 26.70 | 12.81 | 14.70 | 21 |
| LightGBM | Broad (100 trials) | 26.58 | 12.78 | 2.60 | 93 |
| LightGBM | Focused (50 trials) | 26.51 | 12.77 | 2.95 | 45 |
| XGBoost | Broad (100 trials) | 26.69 | 12.82 | 7.47 | 73 |
| XGBoost | Focused (50 trials) | 26.63 | 12.83 | 9.44 | 45 |

**Key Observations:**

- **Best Performance:** LightGBM achieved the lowest MAE (26.51s) across all 450 trials
- **Training Efficiency:** LightGBM training time (2.60-2.95s) significantly faster than CatBoost (11.56-14.70s) and XGBoost (7.47-9.44s), and also a lot more consistent across trials, especially compared to CatBoost
- **Phase Convergence:** Phase 2 focused search provided minimal improvements over Phase 1, indicating effective initial parameter space coverage
- **Model Stability:** LightGBM showed most consistent performance across trials; CatBoost showed highest variability

## Final Model
LightGBM with the following optimized hyperparameters was selected as the final model:

| Parameter | Value |
|-----------|-------|
| `max_depth` | 14 |
| `n_estimators` | 1100 |
| `num_leaves` | 104 |
| `learning_rate` | 0.043 |
| `subsample` | 0.958 |
| `colsample_bytree` | 0.692 |
| `min_child_samples` | 38 |
| `min_split_gain` | 1.2e-05 |
| `reg_alpha` | 6.7e-06 |
| `reg_lambda` | 3.5e-05 |

**Selection Rationale:**

- **Accuracy:** Best MAE (26.51s) across all 450 tuning trials
- **Speed:** Training time ~3x faster than XGBoost and ~5x faster than CatBoost
- **Efficiency:** Strong performance with minimal computational cost, ideal for retraining scenarios
- **Consistency:** Low variance across cross-validation folds, indicating robust generalization
