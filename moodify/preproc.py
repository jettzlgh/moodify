import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def pipeline_scalers(X: pd.DataFrame):
    # Standard scaler = tempo
    # Robust Scaler = loudness
    # MinMax Scaler = key, time_signature

    # Build the pipeline with the different steps
    standard = Pipeline([
        ('standard_scaler', StandardScaler())
    ])
    robust = Pipeline([
        ('robust_scaler', RobustScaler())
    ])
    minmax = Pipeline([
        ('minmax_scaler', MinMaxScaler())
    ])

    # Parallelize the 3 scalers
    preprocessor = ColumnTransformer([
        ('standard', standard, ['tempo']),
        ('robust', robust, ['loudness']),
        ('minmax', minmax, ['key','time_signature'])],
        remainder='passthrough')

    X_scaled = preprocessor.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns = X.columns)
    return X_scaled_df
