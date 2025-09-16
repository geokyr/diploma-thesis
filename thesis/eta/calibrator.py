"""Train-calibrated feature builder for ETA experiments and online inference."""

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from thesis.common.config import (
    N_CLUSTERS,
    N_COMPONENTS,
    PERCENTILE_THRESHOLDS,
    RANDOM_SEED_DEFAULT,
)
from thesis.eta.features import (
    add_all_features,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
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

    def fit(
        self,
        df: pd.DataFrame,
        percentile_thresholds: list[int] = PERCENTILE_THRESHOLDS,
        n_clusters: int = N_CLUSTERS,
        random_seed: int = RANDOM_SEED_DEFAULT,
        n_components: int = N_COMPONENTS,
    ) -> "FeatureCalibrator":
        """
        Fit calibration using training trips only.

        Args:
            df (pd.DataFrame): DataFrame with training trips.
            percentile_thresholds (list[int]): Percentiles to use for distance features.
            n_clusters (int): Number of clusters for K-means clustering on coordinates.
            random_seed (int): Random seed for clustering and pca.
            n_components (int): Number of components for pca.

        Returns:
            FeatureCalibrator: Fitted FeatureCalibrator.

        Raises:
            ValueError: If the required columns are not found in the DataFrame.
        """
        required_columns = ["source_x", "source_y", "destination_x", "destination_y", "distance"]
        if not all(column in df.columns for column in required_columns):
            missing_columns = [column for column in required_columns if column not in df.columns]
            logger.warning(f"Missing required columns for calibration: {missing_columns}")
            return self

        self.distance_percentiles = np.percentile(df["distance"], percentile_thresholds)

        x_center = (df["source_x"] + df["destination_x"]) / 2
        y_center = (df["source_y"] + df["destination_y"]) / 2
        self.city_center_x = np.mean(x_center)
        self.city_center_y = np.mean(y_center)

        source_coordinates = df[["source_x", "source_y"]].values
        destination_coordinates = df[["destination_x", "destination_y"]].values
        self.kmeans_source = KMeans(n_clusters=n_clusters, random_state=random_seed)
        self.kmeans_destination = KMeans(n_clusters=n_clusters, random_state=random_seed)
        self.kmeans_source.fit(source_coordinates)
        self.kmeans_destination.fit(destination_coordinates)

        all_coordinates = np.vstack([source_coordinates, destination_coordinates])
        self.pca_coordinates = PCA(n_components=n_components, random_state=random_seed)
        self.pca_coordinates.fit(all_coordinates)

        logger.info("FeatureCalibrator fitted")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply calibrated feature pipeline to trips.

        Args:
            df (pd.DataFrame): DataFrame with trip data.

        Returns:
            pd.DataFrame: DataFrame with added calibrated features.

        Raises:
            RuntimeError: If the FeatureCalibrator is not fitted.
        """
        self._assert_fitted()

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
        calibrator_path = misc_dir / "feature_calibrator.joblib"
        joblib.dump(self, calibrator_path)

        logger.info(f"Saved FeatureCalibrator to {calibrator_path}")

    @staticmethod
    def load(misc_dir: Path) -> "FeatureCalibrator":
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
        calibrator_path = misc_dir / "feature_calibrator.joblib"

        if not calibrator_path.exists():
            error_msg = f"FeatureCalibrator file not found: {calibrator_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        feature_calibrator = joblib.load(calibrator_path)
        if not isinstance(feature_calibrator, FeatureCalibrator):
            error_msg = "Loaded object is not a FeatureCalibrator"
            logger.error(error_msg)
            raise TypeError(error_msg)

        logger.info(f"Loaded FeatureCalibrator from {calibrator_path}")

        return feature_calibrator

    def _assert_fitted(self) -> None:
        """
        Assert that the FeatureCalibrator is fitted.

        Raises:
            RuntimeError: If the FeatureCalibrator is not fitted.
        """
        if (
            self.distance_percentiles is None
            or self.city_center_x is None
            or self.city_center_y is None
            or self.kmeans_source is None
            or self.kmeans_destination is None
            or self.pca_coordinates is None
        ):
            error_msg = "FeatureCalibrator is not fitted"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
