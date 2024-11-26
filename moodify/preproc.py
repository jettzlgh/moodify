import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def clean_data(df):
    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Word count filter
    df['word_count'] = df['lyrics'].apply(lambda x: len(x.split()))
    df = df[(df['word_count'] < 550) & (df['word_count'] > 40)]

    # Spotify data missing (zero =  missing)
    df  = df[df['energy'] != 0] # 438k

    # RESULT = 438708 lignes
    return df


def preproc_features(df: pd.DataFrame):
    # remove 'speechiness', 'id', 'lyrics', 'is_english', 'genres_list', 'popularity',
    # 'release_date', 'artist_id', 'artist_name', 'artist_popularity', 'artist_followers',
    # 'artist_picture_url', 'liveness', 'type', 'uri', 'track_href', 'analysis_url', 'duration_ms'
    X = df[['danceability',
        'energy',
        'key',
        'loudness',
        'mode',
        'acousticness',
        'instrumentalness',
        'valence',
        'tempo',
        'time_signature'
        ]]

    # drop NaN (2% sur chaque features)
    X = X.dropna()

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
