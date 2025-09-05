"""
Preprocessing pipeline that wraps thesis.eta.features functionality.
"""

import logging
from typing import List

import pandas as pd

from thesis.eta.features import add_all_features

logger = logging.getLogger(__name__)


def preprocess_trip_batch(trip_data: List[dict]) -> pd.DataFrame:
    """
    Convert trip data to features using existing thesis pipeline.

    Args:
        trip_data: List of trip dictionaries with keys:
            - source_x, source_y, destination_x, destination_y
            - time_start, distance

    Returns:
        DataFrame with all features applied (ready for model prediction)
    """
    try:
        # Convert to DataFrame
        df_trips = pd.DataFrame(trip_data)

        if df_trips.empty:
            logger.warning("Empty trip batch received")
            return df_trips

        logger.debug(f"Processing batch of {len(df_trips)} trips")

        # Apply existing feature engineering from thesis.eta.features
        df_features = add_all_features(df_trips)

        logger.debug(f"Generated {len(df_features.columns)} features for {len(df_features)} trips")

        return df_features

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise


def validate_trip_data(trip_data: List[dict]) -> bool:
    """Validate that trip data has required fields."""
    required_fields = ["source_x", "source_y", "destination_x", "destination_y", "time_start", "distance"]

    if not trip_data:
        return False

    for trip in trip_data:
        if not all(field in trip for field in required_fields):
            logger.error(f"Trip missing required fields: {trip}")
            return False

        # Basic validation
        for field in required_fields:
            if trip[field] is None:
                logger.error(f"Trip has null value for {field}")
                return False

    return True


def prepare_features_for_prediction(df_features: pd.DataFrame, target_columns: List[str] = None) -> pd.DataFrame:
    """
    Prepare features for model prediction by removing target columns if present.

    Args:
        df_features: DataFrame with all features
        target_columns: List of target column names to remove (default: ['duration'])

    Returns:
        DataFrame ready for model.predict()
    """
    if target_columns is None:
        target_columns = ["duration"]

    # Remove target columns if they exist
    columns_to_remove = [col for col in target_columns if col in df_features.columns]

    if columns_to_remove:
        df_prediction_ready = df_features.drop(columns_to_remove, axis=1)
        logger.debug(f"Removed {len(columns_to_remove)} target columns for prediction")
    else:
        df_prediction_ready = df_features

    return df_prediction_ready
