# Notes

## To Do
- Fix the results overview notebook

- Sklearn pipeline for preprocessing and training
- Scaling of new features
- Log/Quantile/Box-Cox transformations of new features
- Training with various sub-trip augmentation rates

- One hot encode the hour bin
- Feature engineering
- Feature selection and feature importance
- SHAP values
- Extra distances (euclidean, manhattan, heaversine)
- Center and diff of lat and long coordinates
- Clustering features like MiniKBatchMeans for start and end coordinates, PCA for all coordinates
- Grids/cells/binning/clusters

- Hyperparameter tuning with optuna
- Final model
- Change colorscheme

```python
# Spatial Binning/Grid Encoding
def add_grid_features(df, grid_size=100):
    """Divide space into grid cells"""
    df['source_grid_x'] = (df['source_x'] // grid_size).astype(int)
    df['source_grid_y'] = (df['source_y'] // grid_size).astype(int)
    df['dest_grid_x'] = (df['destination_x'] // grid_size).astype(int)
    df['dest_grid_y'] = (df['destination_y'] // grid_size).astype(int)
    
    # One-hot encode grid cells or use as categorical
    return df

# mRelative Positioning
def add_relative_features(df):
    """Use relative positions instead of absolute"""
    df['delta_x'] = df['destination_x'] - df['source_x']
    df['delta_y'] = df['destination_y'] - df['source_y']
    df['angle'] = np.arctan2(df['delta_y'], df['delta_x'])
    df['euclidean_distance'] = np.sqrt(df['delta_x']**2 + df['delta_y']**2)
    return df

# Coordinate Normalization
def normalize_coordinates(df):
    """Normalize to [0,1] range or z-score"""
    scaler = StandardScaler()
    df[['source_x_norm', 'source_y_norm']] = scaler.fit_transform(
        df[['source_x', 'source_y']]
    )
    return df

from sklearn.cluster import KMeans

# Clustering-Based Encoding
def cluster_encode_locations(df, n_clusters=50):
    """Group similar starting points"""
    kmeans = KMeans(n_clusters=n_clusters)
    df['source_cluster'] = kmeans.fit_predict(df[['source_x', 'source_y']])
    # Use cluster centers as features
    return df

# Add Random Spawn Offsets
def add_spawn_noise(df, noise_std=50):
    """Add Gaussian noise to spawn coordinates"""
    df['source_x_noisy'] = df['source_x'] + np.random.normal(0, noise_std, len(df))
    df['source_y_noisy'] = df['source_y'] + np.random.normal(0, noise_std, len(df))
    return df

# Recommended Combination Approach
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def enhance_trip_features(df_trips, n_clusters=50, grid_size=100):
    """
    Comprehensive feature engineering to handle limited source coordinate diversity.
    
    Args:
        df_trips: DataFrame with source_x, source_y, destination_x, destination_y
        n_clusters: Number of clusters for location encoding
        grid_size: Size of grid cells in meters
    
    Returns:
        DataFrame with enhanced features
    """
    df = df_trips.copy()
    
    # 1. Relative positioning features (most important!)
    df['delta_x'] = df['destination_x'] - df['source_x']
    df['delta_y'] = df['destination_y'] - df['source_y']
    df['euclidean_distance'] = np.sqrt(df['delta_x']**2 + df['delta_y']**2)
    df['manhattan_distance'] = np.abs(df['delta_x']) + np.abs(df['delta_y'])
    df['direction_angle'] = np.arctan2(df['delta_y'], df['delta_x'])
    
    # 2. Grid-based encoding
    df['source_grid_x'] = (df['source_x'] // grid_size).astype(int)
    df['source_grid_y'] = (df['source_y'] // grid_size).astype(int)
    df['dest_grid_x'] = (df['destination_x'] // grid_size).astype(int)
    df['dest_grid_y'] = (df['destination_y'] // grid_size).astype(int)
    
    # Grid distance features
    df['grid_delta_x'] = df['dest_grid_x'] - df['source_grid_x']
    df['grid_delta_y'] = df['dest_grid_y'] - df['source_grid_y']
    df['grid_distance'] = np.sqrt(df['grid_delta_x']**2 + df['grid_delta_y']**2)
    
    # 3. Cluster-based encoding for common start locations
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    source_coords = df[['source_x', 'source_y']].values
    df['source_cluster'] = kmeans.fit_predict(source_coords)
    
    # Distance to cluster center
    cluster_centers = kmeans.cluster_centers_
    source_clusters = df['source_cluster'].values
    df['dist_to_source_cluster_center'] = np.array([
        np.linalg.norm(source_coords[i] - cluster_centers[source_clusters[i]])
        for i in range(len(df))
    ])
    
    # 4. Normalized coordinates (helps with generalization)
    scaler = StandardScaler()
    df[['source_x_norm', 'source_y_norm']] = scaler.fit_transform(df[['source_x', 'source_y']])
    df[['dest_x_norm', 'dest_y_norm']] = scaler.fit_transform(df[['destination_x', 'destination_y']])
    
    # 5. Quadrant-based features
    df['source_quadrant'] = (
        (df['source_x'] > df['source_x'].median()).astype(int) * 2 +
        (df['source_y'] > df['source_y'].median()).astype(int)
    )
    df['dest_quadrant'] = (
        (df['destination_x'] > df['destination_x'].median()).astype(int) * 2 +
        (df['destination_y'] > df['destination_y'].median()).astype(int)
    )
    df['same_quadrant'] = (df['source_quadrant'] == df['dest_quadrant']).astype(int)
    
    # 6. Add noise to source coordinates for training (regularization)
    if 'is_training' in df.columns and df['is_training'].any():
        noise_std = 30  # meters
        df.loc[df['is_training'], 'source_x_noisy'] = (
            df.loc[df['is_training'], 'source_x'] + 
            np.random.normal(0, noise_std, df['is_training'].sum())
        )
        df.loc[df['is_training'], 'source_y_noisy'] = (
            df.loc[df['is_training'], 'source_y'] + 
            np.random.normal(0, noise_std, df['is_training'].sum())
        )
    
    # 7. Ratio features
    df['x_ratio'] = df['delta_x'] / (df['euclidean_distance'] + 1e-6)
    df['y_ratio'] = df['delta_y'] / (df['euclidean_distance'] + 1e-6)
    
    return df


def select_robust_features(df, use_raw_coords=False):
    """
    Select features that are robust to limited source diversity.
    
    Args:
        df: DataFrame with enhanced features
        use_raw_coords: Whether to include raw coordinates (not recommended)
    
    Returns:
        List of feature column names
    """
    # Features that don't depend heavily on exact source location
    robust_features = [
        # Relative features (most important)
        'delta_x', 'delta_y', 'euclidean_distance', 'manhattan_distance',
        'direction_angle', 'x_ratio', 'y_ratio',
        
        # Grid-based features
        'grid_delta_x', 'grid_delta_y', 'grid_distance',
        
        # Cluster and quadrant features
        'source_cluster', 'dist_to_source_cluster_center',
        'source_quadrant', 'dest_quadrant', 'same_quadrant',
        
        # Time features (if available)
        'hour_bin',
        
        # Trip characteristics
        'distance', 'duration'
    ]
    
    if use_raw_coords:
        # Not recommended due to overfitting risk
        robust_features.extend(['source_x_norm', 'source_y_norm', 
                               'dest_x_norm', 'dest_y_norm'])
    
    # Only return features that exist in the dataframe
    return [f for f in robust_features if f in df.columns]


# Example usage
if __name__ == "__main__":
    # Simulated trip data
    trips = pd.DataFrame({
        'source_x': np.random.choice([1000, 2000, 3000], 1000),  # Limited diversity
        'source_y': np.random.choice([500, 1500, 2500], 1000),   # Limited diversity
        'destination_x': np.random.uniform(0, 5000, 1000),       # High diversity
        'destination_y': np.random.uniform(0, 5000, 1000),       # High diversity
        'hour_bin': np.random.randint(0, 24, 1000),
        'distance': np.random.uniform(500, 10000, 1000),
        'duration': np.random.uniform(60, 3600, 1000),
        'is_training': True
    })
    
    # Enhance features
    trips_enhanced = enhance_trip_features(trips)
    
    # Select robust features
    feature_cols = select_robust_features(trips_enhanced)
    print(f"Selected {len(feature_cols)} robust features")
    print(f"Features: {feature_cols}")
```
