"""Feature engineering and transformation utilities for ETA prediction."""

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler

from thesis.common.config import (
    AFTERNOON_FLOOR,
    CELL,
    COORDINATE_SCALE,
    FEATURE_CALIBRATOR_FILENAME,
    MORNING_CEILING,
    N_CLUSTERS,
    N_COMPONENTS,
    NOON_CEILING,
    NOON_FLOOR,
    NUM_FREQS,
    PERCENTILE_THRESHOLDS,
    RANDOM_SEED_DEFAULT,
    RUSH_HOURS,
)
from thesis.eta.models import ModelType

logger = logging.getLogger(__name__)


def check_required_columns(df: pd.DataFrame, required_columns: list[str], feature_type: str) -> None:
    """
    Check if DataFrame contains all required columns.

    Args:
        df (pd.DataFrame): DataFrame to check.
        required_columns (list[str]): List of required column names.
        feature_type (str): Type of features being added (for logging).

    Raises:
        ValueError: If the required columns are not found in the DataFrame.
    """
    if not all(column in df.columns for column in required_columns):
        missing_columns = [column for column in required_columns if column not in df.columns]

        error_msg = f"{missing_columns} not found, while adding {feature_type} features"
        logger.error(error_msg)
        raise ValueError(error_msg)


def _log_feature_addition(df_original: pd.DataFrame, df_final: pd.DataFrame, feature_type: str) -> None:
    """
    Log the number of features added.

    Args:
        df_original (pd.DataFrame): Original DataFrame before adding features.
        df_final (pd.DataFrame): Final DataFrame after adding features.
        feature_type (str): Type of features that were added.
    """
    n_initial_features = len(df_original.columns)
    n_final_features = len(df_final.columns)
    n_added_features = n_final_features - n_initial_features

    logger.info(f"Added {n_added_features} {feature_type} features, for a total of {n_final_features} features")


def split_features_and_target(
    df: pd.DataFrame, target_columns: list[str] = ["duration"]
) -> tuple[pd.DataFrame, pd.DataFrame | pd.Series]:
    """
    Split the features and the target.

    Args:
        df (pd.DataFrame): A DataFrame containing the prepared data.
        target_columns (list[str]): List of column names to use as target, defaults to ["duration"].

    Returns:
        tuple[pd.DataFrame, pd.DataFrame | pd.Series]: A tuple containing the features and the target.

    Raises:
        ValueError: If the DataFrame does not contain the target columns.
    """
    missing_columns = [col for col in target_columns if col not in df.columns]
    if missing_columns:
        error_msg = f"DataFrame must contain target columns: {missing_columns}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    X = df.drop(target_columns, axis=1)
    y = df[target_columns]

    n_targets = len(target_columns)
    if n_targets == 1:
        y = y.iloc[:, 0]

    n_samples = len(df)
    variable_word = "variable" if n_targets == 1 else "variables"
    logger.info(f"Split {n_samples} samples into {len(X.columns)} features and {n_targets} target {variable_word}")
    return X, y


def log_transform_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame | None = None, feature_names: list[str] = ["distance"]
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Log transform features.

    Args:
        X_train (pd.DataFrame): Training features to log transform.
        X_test (pd.DataFrame | None): Test features to log transform.
        feature_names (list[str]): List of feature names to log transform.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame | None]: Log transformed training and optionally test features.
    """
    X_train_log = X_train.copy()
    X_train_log[feature_names] = np.log1p(X_train[feature_names])

    if X_test is not None:
        X_test_log = X_test.copy()
        X_test_log[feature_names] = np.log1p(X_test[feature_names])
        logger.info(f"Log transformed {len(X_train)} training samples and {len(X_test)} test samples")
        return X_train_log, X_test_log
    else:
        logger.info(f"Log transformed {len(X_train)} training samples")
        return X_train_log, None


def standard_scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Standard scale training and optionally test features using StandardScaler.

    Args:
        X_train (pd.DataFrame): Training features to fit scaler on.
        X_test (pd.DataFrame | None): Test features to transform.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame | None]: Scaled training features and optionally scaled test features.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        logger.info(f"Standard scaled {len(X_train)} training samples and {len(X_test)} test samples")
        return X_train_scaled, X_test_scaled
    else:
        logger.info(f"Standard scaled {len(X_train)} training samples")
        return X_train_scaled, None


def create_log_transformer() -> FunctionTransformer:
    """
    Create a log transformer with exponential inverse function.

    Returns:
        FunctionTransformer: Log transformer with exponential inverse function.
    """
    return FunctionTransformer(func=np.log1p, inverse_func=np.expm1, check_inverse=True)


def create_quantile_transformer(random_seed: int = RANDOM_SEED_DEFAULT) -> QuantileTransformer:
    """
    Create a quantile transformer.

    Args:
        random_seed (int): The random seed to use for the random number generator.

    Returns:
        QuantileTransformer: Quantile transformer.
    """
    return QuantileTransformer(output_distribution="normal", random_state=random_seed)


def create_boxcox_transformer() -> PowerTransformer:
    """
    Create a box cox transformer.

    Returns:
        PowerTransformer: Box cox transformer.
    """
    return PowerTransformer(method="box-cox")


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing time_start column.

    Returns:
        pd.DataFrame: DataFrame with added temporal features.

    Raises:
        ValueError: If the required columns are not found in the DataFrame.
    """
    required_columns = ["time_start"]
    check_required_columns(df, required_columns, "temporal")

    df_hour = df.copy()

    df_hour["hour_bin"] = (df_hour["time_start"] % 36000) // 3600
    df_hour["is_morning"] = (df_hour["hour_bin"] <= MORNING_CEILING).astype(int)
    df_hour["is_noon"] = ((df_hour["hour_bin"] >= NOON_FLOOR) & (df_hour["hour_bin"] <= NOON_CEILING)).astype(int)
    df_hour["is_afternoon"] = (df_hour["hour_bin"] >= AFTERNOON_FLOOR).astype(int)
    df_hour["is_rush_hour"] = df_hour["hour_bin"].isin(RUSH_HOURS).astype(int)

    _log_feature_addition(df, df_hour, "temporal")

    return df_hour


def add_spatial_features(
    df: pd.DataFrame,
    distance_percentiles: np.ndarray | None = None,
    percentile_thresholds: list[int] = PERCENTILE_THRESHOLDS,
    city_center_x: float | None = None,
    city_center_y: float | None = None,
) -> pd.DataFrame:
    """
    Add spatial features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing source_x, source_y, destination_x, destination_y, distance columns.
        distance_percentiles (np.ndarray | None): Percentiles to use for distance features.
        percentile_thresholds (list[int]): Percentiles to use for distance features.
        city_center_x (float | None): Mean center x to use for radial features.
        city_center_y (float | None): Mean center y to use for radial features.

    Returns:
        pd.DataFrame: DataFrame with added spatial features.

    Raises:
        ValueError: If the required columns are not found in the DataFrame.
    """
    required_columns = ["source_x", "source_y", "destination_x", "destination_y", "distance"]
    check_required_columns(df, required_columns, "spatial")

    df_spatial = df.copy()

    df_spatial["x_center"] = (df_spatial["source_x"] + df_spatial["destination_x"]) / 2
    df_spatial["y_center"] = (df_spatial["source_y"] + df_spatial["destination_y"]) / 2
    df_spatial["x_difference"] = df_spatial["destination_x"] - df_spatial["source_x"]
    df_spatial["y_difference"] = df_spatial["destination_y"] - df_spatial["source_y"]

    df_spatial["euclidean_distance"] = np.hypot(df_spatial["x_difference"], df_spatial["y_difference"])
    df_spatial["route_efficiency"] = df_spatial["euclidean_distance"] / df_spatial["distance"]
    df_spatial["detour_length"] = df_spatial["distance"] - df_spatial["euclidean_distance"]

    df_spatial["trip_bearing"] = np.arctan2(df_spatial["y_difference"], df_spatial["x_difference"])
    df_spatial["trip_bearing_sin"] = np.sin(df_spatial["trip_bearing"])
    df_spatial["trip_bearing_cos"] = np.cos(df_spatial["trip_bearing"])

    if distance_percentiles is None:
        distance_percentiles = np.percentile(df_spatial["distance"], percentile_thresholds)

    df_spatial["is_short_distance"] = (df_spatial["distance"] <= distance_percentiles[0]).astype(int)
    df_spatial["is_medium_distance"] = (
        (df_spatial["distance"] > distance_percentiles[0]) & (df_spatial["distance"] <= distance_percentiles[2])
    ).astype(int)
    df_spatial["is_long_distance"] = (df_spatial["distance"] > distance_percentiles[2]).astype(int)

    if city_center_x is None:
        city_center_x = np.mean(df_spatial["x_center"])
    if city_center_y is None:
        city_center_y = np.mean(df_spatial["y_center"])

    df_spatial["source_distance_from_city_center"] = np.hypot(
        df_spatial["source_x"] - city_center_x, df_spatial["source_y"] - city_center_y
    )
    df_spatial["dest_distance_from_city_center"] = np.hypot(
        df_spatial["destination_x"] - city_center_x, df_spatial["destination_y"] - city_center_y
    )
    df_spatial["trip_centrality_change"] = (
        df_spatial["dest_distance_from_city_center"] - df_spatial["source_distance_from_city_center"]
    )
    df_spatial["trip_centrality"] = np.hypot(
        df_spatial["x_center"] - city_center_x, df_spatial["y_center"] - city_center_y
    )

    _log_feature_addition(df, df_spatial, "spatial")

    return df_spatial


def add_fourier_features(
    df: pd.DataFrame, coordinate_scale: float = COORDINATE_SCALE, num_freqs: int = NUM_FREQS
) -> pd.DataFrame:
    """
    Add fourier positional encoding features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing source_x, source_y, destination_x, destination_y columns.
        coordinate_scale (float): Scale factor to normalize coordinates for fourier encoding.
        num_freqs (int): Number of frequency components to use for fourier encoding.

    Returns:
        pd.DataFrame: DataFrame with added fourier features.

    Raises:
        ValueError: If the required columns are not found in the DataFrame.
    """
    required_columns = ["source_x", "source_y", "destination_x", "destination_y"]
    check_required_columns(df, required_columns, "fourier")

    df_fourier = df.copy()

    coordinate_columns = ["source_x", "source_y", "destination_x", "destination_y"]
    for column in coordinate_columns:
        normalized_coordinates = df_fourier[column] / coordinate_scale

        freqs = np.array([2**i for i in range(num_freqs)])

        for i, freq in enumerate(freqs):
            df_fourier[f"{column}_sin_{i}"] = np.sin(freq * normalized_coordinates)
            df_fourier[f"{column}_cos_{i}"] = np.cos(freq * normalized_coordinates)

    _log_feature_addition(df, df_fourier, "fourier")

    return df_fourier


def add_cell_features(df: pd.DataFrame, cell: int = CELL) -> pd.DataFrame:
    """
    Add cell features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing source_x, source_y, destination_x, destination_y columns.
        cell (int): Size of the cell in meters.

    Returns:
        pd.DataFrame: DataFrame with added cell features.

    Raises:
        ValueError: If the required columns are not found in the DataFrame.
    """
    required_columns = ["source_x", "source_y", "destination_x", "destination_y"]
    check_required_columns(df, required_columns, "cell")

    df_cell = df.copy()

    df_cell["source_cell_x"] = (df_cell["source_x"] // cell).astype(int)
    df_cell["source_cell_y"] = (df_cell["source_y"] // cell).astype(int)
    df_cell["destination_cell_x"] = (df_cell["destination_x"] // cell).astype(int)
    df_cell["destination_cell_y"] = (df_cell["destination_y"] // cell).astype(int)

    _log_feature_addition(df, df_cell, "cell")

    return df_cell


def add_clustering_features(
    df: pd.DataFrame,
    kmeans_source: KMeans | None = None,
    kmeans_destination: KMeans | None = None,
    n_clusters: int = N_CLUSTERS,
    random_seed: int = RANDOM_SEED_DEFAULT,
) -> pd.DataFrame:
    """
    Add clustering features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing source_x, source_y, destination_x, destination_y columns.
        kmeans_source (KMeans | None): KMeans object to use for source clustering.
        kmeans_destination (KMeans | None): KMeans object to use for destination clustering.
        n_clusters (int): Number of clusters for K-means clustering on coordinates.
        random_seed (int): Random seed for clustering.

    Returns:
        pd.DataFrame: DataFrame with added clustering features.

    Raises:
        ValueError: If the required columns are not found in the DataFrame.
    """
    required_columns = ["source_x", "source_y", "destination_x", "destination_y"]
    check_required_columns(df, required_columns, "clustering")

    df_cluster = df.copy()

    source_coordinates = df_cluster[["source_x", "source_y"]].values
    destination_coordinates = df_cluster[["destination_x", "destination_y"]].values
    if kmeans_source is None:
        kmeans_source = KMeans(n_clusters=n_clusters, random_state=random_seed)
        df_cluster["source_cluster"] = kmeans_source.fit_predict(source_coordinates)
    else:
        df_cluster["source_cluster"] = kmeans_source.predict(source_coordinates)

    if kmeans_destination is None:
        kmeans_destination = KMeans(n_clusters=n_clusters, random_state=random_seed)
        df_cluster["destination_cluster"] = kmeans_destination.fit_predict(destination_coordinates)
    else:
        df_cluster["destination_cluster"] = kmeans_destination.predict(destination_coordinates)

    _log_feature_addition(df, df_cluster, "clustering")

    return df_cluster


def add_pca_features(
    df: pd.DataFrame,
    pca_coordinates: PCA | None = None,
    n_components: int = N_COMPONENTS,
    random_seed: int = RANDOM_SEED_DEFAULT,
) -> pd.DataFrame:
    """
    Add pca features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing source_x, source_y, destination_x, destination_y columns.
        pca_coordinates (PCA | None): PCA object to use for pca.
        n_components (int): Number of components for pca.
        random_seed (int): Random seed for pca.

    Returns:
        pd.DataFrame: DataFrame with added pca features.

    Raises:
        ValueError: If the required columns are not found in the DataFrame.
    """
    required_columns = ["source_x", "source_y", "destination_x", "destination_y"]
    check_required_columns(df, required_columns, "pca")

    df_pca = df.copy()

    source_coordinates = df_pca[["source_x", "source_y"]].values
    destination_coordinates = df_pca[["destination_x", "destination_y"]].values

    if pca_coordinates is None:
        all_coordinates = np.vstack([source_coordinates, destination_coordinates])
        pca_coordinates = PCA(n_components=n_components, random_state=random_seed)
        pca_coordinates.fit(all_coordinates)

    source_coordinates_pca = pca_coordinates.transform(source_coordinates)
    destination_coordinates_pca = pca_coordinates.transform(destination_coordinates)

    df_pca["source_pca_1"] = source_coordinates_pca[:, 0]
    df_pca["source_pca_2"] = source_coordinates_pca[:, 1]
    df_pca["dest_pca_1"] = destination_coordinates_pca[:, 0]
    df_pca["dest_pca_2"] = destination_coordinates_pca[:, 1]

    _log_feature_addition(df, df_pca, "pca")

    return df_pca


def add_all_features(
    df: pd.DataFrame,
    distance_percentiles: np.ndarray | None = None,
    percentile_thresholds: list[int] = PERCENTILE_THRESHOLDS,
    city_center_x: float | None = None,
    city_center_y: float | None = None,
    coordinate_scale: float = COORDINATE_SCALE,
    num_freqs: int = NUM_FREQS,
    cell: int = CELL,
    kmeans_source: KMeans | None = None,
    kmeans_destination: KMeans | None = None,
    n_clusters: int = N_CLUSTERS,
    random_seed: int = RANDOM_SEED_DEFAULT,
    pca_coordinates: PCA | None = None,
    n_components: int = N_COMPONENTS,
) -> pd.DataFrame:
    """
    Add all features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing source_x, source_y, destination_x, destination_y, distance columns.
        distance_percentiles (np.ndarray | None): Percentiles to use for distance features.
        percentile_thresholds (list[int]): Percentiles to use for distance features.
        city_center_x (float | None): Mean center x to use for radial features.
        city_center_y (float | None): Mean center y to use for radial features.
        coordinate_scale (float): Scale factor to normalize coordinates for fourier encoding.
        num_freqs (int): Number of frequency components to use for fourier encoding.
        cell (int): Size of the cell in meters.
        kmeans_source (KMeans | None): KMeans object to use for source clustering.
        kmeans_destination (KMeans | None): KMeans object to use for destination clustering.
        n_clusters (int): Number of clusters for K-means clustering on coordinates.
        random_seed (int): Random seed for clustering and pca.
        pca_coordinates (PCA | None): PCA object to use for pca.
        n_components (int): Number of components for pca.

    Returns:
        pd.DataFrame: DataFrame with added all features.

    Raises:
        ValueError: If the required columns are not found in the DataFrame.
    """
    df = add_temporal_features(df)
    df = add_spatial_features(df, distance_percentiles, percentile_thresholds, city_center_x, city_center_y)
    df = add_fourier_features(df, coordinate_scale, num_freqs)
    df = add_cell_features(df, cell)
    df = add_clustering_features(df, kmeans_source, kmeans_destination, n_clusters, random_seed)
    df = add_pca_features(df, pca_coordinates, n_components, random_seed)

    return df


def identify_categorical_features(df: pd.DataFrame) -> list[str]:
    """
    Identify categorical features in the dataset based on feature names and characteristics.

    Args:
        df (pd.DataFrame): DataFrame with trip data.

    Returns:
        list[str]: List of categorical feature column names.
    """
    categorical_features = []

    for col in df.columns:
        if "cluster" in col:
            categorical_features.append(col)
        elif "cell" in col:
            categorical_features.append(col)
        elif "bin" in col:
            categorical_features.append(col)
        elif col.startswith("is_") and df[col].dtype in ["int64", "int32"]:
            unique_vals = set(df[col].unique())
            if unique_vals.issubset({0, 1}):
                categorical_features.append(col)

    return categorical_features


def prepare_features_for_catboost(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """
    Prepare features for CatBoost.

    Args:
        df (pd.DataFrame): DataFrame with trip data.

    Returns:
        tuple[pd.DataFrame, list[int]]: Tuple of prepared DataFrame and list of categorical feature indices.
    """
    df_prepared = df.copy()
    categorical_features = identify_categorical_features(df_prepared)

    categorical_indices = []
    for col in categorical_features:
        if col in df_prepared.columns:
            categorical_indices.append(df_prepared.columns.get_loc(col))

    logger.info(f"{ModelType.CATBOOST_REGRESSOR}: Identified {len(categorical_indices)} categorical features")
    return df_prepared, categorical_indices


def prepare_features_for_xgboost_lightgbm(df: pd.DataFrame, model_type: ModelType) -> pd.DataFrame:
    """
    Prepare features for XGBoost and LightGBM.

    Args:
        df (pd.DataFrame): DataFrame with trip data.
        model_type (ModelType): Type of model for logging purposes.

    Returns:
        pd.DataFrame: Prepared DataFrame.
    """
    df_prepared = df.copy()
    categorical_features = identify_categorical_features(df_prepared)

    for col in categorical_features:
        if col in df_prepared.columns:
            df_prepared[col] = df_prepared[col].astype("category")

    logger.info(f"{model_type}: Converted {len(categorical_features)} features to category dtype")
    return df_prepared


def optimize_features_for_model(df: pd.DataFrame, model_type: ModelType) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """
    Optimize features for a specific model and return appropriate fit_kwargs.

    Args:
        df (pd.DataFrame): DataFrame with trip data.
        model_type (ModelType): Type of model.

    Returns:
        tuple[pd.DataFrame, dict[str, list[int]]]: Tuple of optimized DataFrame and fit_kwargs for the model.
    """
    if model_type == ModelType.CATBOOST_REGRESSOR:
        df_opt, cat_indices = prepare_features_for_catboost(df)
        fit_kwargs = {"cat_features": cat_indices} if cat_indices else {}
        return df_opt, fit_kwargs

    elif model_type in (ModelType.XGBOOST_REGRESSOR, ModelType.LIGHTGBM_REGRESSOR):
        df_opt = prepare_features_for_xgboost_lightgbm(df, model_type)
        return df_opt, {}

    else:
        logger.info(f"{model_type}: No specific optimization")
        return df.copy(), {}


@dataclass(frozen=True, slots=True)
class FeatureCalibrator:
    """
    Feature calibrator to fit on train trips and transform test and rain trips.

    Attributes:
        distance_percentiles (np.ndarray): thresholds at configured percentiles
        city_center_x (float): mean center x used for radial features
        city_center_y (float): mean center y used for radial features
        kmeans_source (KMeans): clustering on source coordinates
        kmeans_destination (KMeans): clustering on destination coordinates
        pca_coordinates (PCA): PCA on coordinates
    """

    distance_percentiles: np.ndarray
    city_center_x: float
    city_center_y: float
    kmeans_source: KMeans
    kmeans_destination: KMeans
    pca_coordinates: PCA

    @classmethod
    def from_train_trips(
        cls,
        df: pd.DataFrame,
        percentile_thresholds: list[int] = PERCENTILE_THRESHOLDS,
        n_clusters: int = N_CLUSTERS,
        random_seed: int = RANDOM_SEED_DEFAULT,
        n_components: int = N_COMPONENTS,
    ) -> "FeatureCalibrator":
        """
        Create and fit a FeatureCalibrator from training trips.

        Args:
            df (pd.DataFrame): DataFrame with training trips.
            percentile_thresholds (list[int]): Percentiles to use for distance features.
            n_clusters (int): Number of clusters for K-means clustering on coordinates.
            random_seed (int): Random seed for clustering and pca.
            n_components (int): Number of components for pca.

        Returns:
            FeatureCalibrator: New fitted FeatureCalibrator.

        Raises:
            ValueError: If the required columns are not found in the DataFrame.
        """
        required_columns = ["source_x", "source_y", "destination_x", "destination_y", "distance"]
        check_required_columns(df, required_columns, "calibration")

        distance_percentiles = np.percentile(df["distance"], percentile_thresholds)

        x_center = (df["source_x"] + df["destination_x"]) / 2
        y_center = (df["source_y"] + df["destination_y"]) / 2
        city_center_x = np.mean(x_center)
        city_center_y = np.mean(y_center)

        source_coordinates = df[["source_x", "source_y"]].values
        destination_coordinates = df[["destination_x", "destination_y"]].values
        kmeans_source = KMeans(n_clusters=n_clusters, random_state=random_seed)
        kmeans_destination = KMeans(n_clusters=n_clusters, random_state=random_seed)
        kmeans_source.fit(source_coordinates)
        kmeans_destination.fit(destination_coordinates)

        all_coordinates = np.vstack([source_coordinates, destination_coordinates])
        pca_coordinates = PCA(n_components=n_components, random_state=random_seed)
        pca_coordinates.fit(all_coordinates)

        logger.info("New FeatureCalibrator fitted")

        return cls(
            distance_percentiles=distance_percentiles,
            city_center_x=city_center_x,
            city_center_y=city_center_y,
            kmeans_source=kmeans_source,
            kmeans_destination=kmeans_destination,
            pca_coordinates=pca_coordinates,
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply calibrated feature pipeline to trips.

        Args:
            df (pd.DataFrame): DataFrame with trip data.

        Returns:
            pd.DataFrame: DataFrame with added calibrated features.

        Raises:
            ValueError: If the required columns are not found in the DataFrame.
        """
        df = add_all_features(
            df,
            distance_percentiles=self.distance_percentiles,
            city_center_x=self.city_center_x,
            city_center_y=self.city_center_y,
            kmeans_source=self.kmeans_source,
            kmeans_destination=self.kmeans_destination,
            pca_coordinates=self.pca_coordinates,
        )

        return df

    def save(self, misc_dir: Path) -> None:
        """
        Save the FeatureCalibrator to the misc directory.

        Args:
            misc_dir (Path): Directory to save the FeatureCalibrator to.
        """
        calibrator_path = misc_dir / FEATURE_CALIBRATOR_FILENAME
        joblib.dump(self, calibrator_path)

        logger.info(f"FeatureCalibrator saved to {calibrator_path}")

    @classmethod
    def load(cls, misc_dir: Path) -> "FeatureCalibrator":
        """
        Load the FeatureCalibrator from the misc directory.

        Args:
            misc_dir (Path): Directory to load the FeatureCalibrator from.

        Returns:
            FeatureCalibrator: Loaded FeatureCalibrator.

        Raises:
            FileNotFoundError: If the FeatureCalibrator file does not exist.
            TypeError: If the loaded object is not a FeatureCalibrator.
        """
        calibrator_path = misc_dir / FEATURE_CALIBRATOR_FILENAME
        if not calibrator_path.exists():
            error_msg = f"FeatureCalibrator file not found: {calibrator_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        feature_calibrator = joblib.load(calibrator_path)
        if not isinstance(feature_calibrator, cls):
            error_msg = f"Loaded object is not a {cls.__name__}"
            logger.error(error_msg)
            raise TypeError(error_msg)

        logger.info(f"FeatureCalibrator loaded from {calibrator_path}")

        return feature_calibrator
