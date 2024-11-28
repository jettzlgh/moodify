import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

######################## PREPROC AUDIO FEATURES ########################


def clean_data(df):
    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Word count filter
    if 'lyrics' in df.keys():
        df['word_count'] = df['lyrics'].apply(lambda x: len(x.split()))
        df = df[(df['word_count'] < 550) & (df['word_count'] > 40)]

    # Spotify data missing (zero =  missing)
    df  = df[df['energy'] != 0] # 438k

    # drop NaN (2% sur chaque features
    df.dropna(subset=['danceability',
        'energy',
        'key',
        'loudness',
        'mode',
        'acousticness',
        'instrumentalness',
        'valence',
        'tempo',
        'time_signature'
        ], inplace=True)

    df.reset_index(inplace=True)
    # RESULT = 429 437 lignes
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

    # Create the new column names
    transformed_columns = ['tempo', 'loudness', 'key', 'time_signature']
    passthrough_columns = [col for col in X.columns if col not in ['tempo', 'loudness', 'key', 'time_signature']]
    final_columns = transformed_columns + passthrough_columns

    # Conversion en DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=final_columns)

    #429 437 lignes
    return X_scaled_df


def feature_eng(X: pd.DataFrame):
    '''
    Adds transformed features to the sportify dataset based on the emotion different scales:
    - Step 1: adjectives associated to each scale were grouped by level of emotion (sad, neutral and happy - 1 to 3)
    - Step 2: adjectives were grouped by level of energy (low, normal and high - 1 to 3)

    Source: https://ledgernote.com/blog/interesting/musical-key-characteristics-emotions/

    '''

    # Load the hard-coded key and mode emotion and energy scores
    music_dict = [
    {"key": 8, "mode": 1, "adjectives": ["Death", "Eternity", "Judgement"], "emotion_rank": 1, "energy_rank": 1},
    {"key": 6, "mode": 0, "adjectives": ["Gloomy", "Passionate Resentment"], "emotion_rank": 1, "energy_rank": 1},
    {"key": 1, "mode": 1, "adjectives": ["Grief", "Depressive"], "emotion_rank": 1, "energy_rank": 1},
    {"key": 8, "mode": 0, "adjectives": ["Grumbling", "Moaning", "Wailing"], "emotion_rank": 2, "energy_rank": 1},
    {"key": 5, "mode": 0, "adjectives": ["Obscure", "Plaintive", "Funereal"], "emotion_rank": 1, "energy_rank": 1},
    {"key": 11, "mode": 0, "adjectives": ["Solitary", "Melancholic", "Patience"], "emotion_rank": 2, "energy_rank": 1},
    {"key": 9, "mode": 0, "adjectives": ["Tender", "Plaintive", "Pious"], "emotion_rank": 3, "energy_rank": 1},
    {"key": 6, "mode": 1, "adjectives": ["Conquering Difficulties", "Sighs of Relief"], "emotion_rank": 3, "energy_rank": 2},
    {"key": 3, "mode": 1, "adjectives": ["Cruel", "Hard", "Yet Full of Devotion"], "emotion_rank": 2, "energy_rank": 2},
    {"key": 3, "mode": 0, "adjectives": ["Deep Distress", "Existential Angst"], "emotion_rank": 1, "energy_rank": 2},
    {"key": 7, "mode": 0, "adjectives": ["Discontent", "Uneasiness"], "emotion_rank": 2, "energy_rank": 2},
    {"key": 0, "mode": 1, "adjectives": ["Innocently Happy"], "emotion_rank": 3, "energy_rank": 2},
    {"key": 0, "mode": 0, "adjectives": ["Innocently Sad", "Love-Sick"], "emotion_rank": 2, "energy_rank": 2},
    {"key": 9, "mode": 1, "adjectives": ["Joyful", "Pastoral", "Declaration of Love"], "emotion_rank": 3, "energy_rank": 2},
    {"key": 10, "mode": 1, "adjectives": ["Joyful", "Quaint", "Cheerful"], "emotion_rank": 3, "energy_rank": 2},
    {"key": 7, "mode": 1, "adjectives": ["Serious", "Magnificent", "Fantasy"], "emotion_rank": 3, "energy_rank": 2},
    {"key": 2, "mode": 0, "adjectives": ["Serious", "Pious", "Ruminating"], "emotion_rank": 2, "energy_rank": 2},
    {"key": 10, "mode": 0, "adjectives": ["Terrible", "the Night", "Mocking"], "emotion_rank": 1, "energy_rank": 2},
    {"key": 1, "mode": 0, "adjectives": ["Despair", "Wailing", "Weeping"], "emotion_rank": 1, "energy_rank": 3},
    {"key": 4, "mode": 0, "adjectives": ["Effeminate", "Amorous", "Restless"], "emotion_rank": 3, "energy_rank": 3},
    {"key": 5, "mode": 1, "adjectives": ["Furious", "Quick-Tempered", "Passing Regret"], "emotion_rank": 1, "energy_rank": 3},
    {"key": 11, "mode": 1, "adjectives": ["Harsh", "Strong", "Wild", "Rage"], "emotion_rank": 1, "energy_rank": 3},
    {"key": 4, "mode": 1, "adjectives": ["Quarrelsome", "Boisterous", "Incomplete Pleasure"], "emotion_rank": 3, "energy_rank": 3},
    {"key": 2, "mode": 1, "adjectives": ["Triumphant", "Victorious War-Cries"], "emotion_rank": 3, "energy_rank": 3}
    ]

    music_df = pd.DataFrame(music_dict)

    # Drop adjectives
    music_df.drop(columns=['adjectives'], inplace=True)

    # Merge emotion and energy rank into the main data
    df = pd.merge(X,music_df, how='left', left_on= ['key', 'mode'], right_on = ['key', 'mode'])

    # Atention! This feature is being tested
    # X['emotion_energy_rank'] = X['emotion_rank'] * X['energy_rank']

    return df


def preproc_features_two(df: pd.DataFrame):
    '''
    Does the same thing as preproc_features, but also encodes emotion_rank and energy_rank using one hote encoder.
    '''
    # remove 'speechiness', 'id', 'lyrics', 'is_english', 'genres_list', 'popularity',
    # 'release_date', 'artist_id', 'artist_name', 'artist_popularity', 'artist_followers',
    # 'artist_picture_url', 'liveness', 'type', 'uri', 'track_href', 'analysis_url', 'duration_ms'

    X = df[['danceability',
        'energy',
        'loudness',
        'acousticness',
        'instrumentalness',
        'valence',
        'time_signature',
        'emotion_rank',
        'energy_rank',
        'tempo'
        ]]

    # Standard scaler = tempo
    # Robust Scaler = loudness
    # MinMax Scaler = time_signature
    # OneHotEncoder Scaler = emotion_rank, energy_rank

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
    onehot = Pipeline([
        ('onehot', OneHotEncoder(drop='first'))
])

    # Parallelize the 3 scalers
    preprocessor = ColumnTransformer([
        ('standard', standard, ['tempo']),
        ('robust', robust, ['loudness']),
        ('minmax', minmax, ['time_signature']),
        ('onehot', onehot, ['emotion_rank','energy_rank'])],
        remainder='passthrough')

    X_scaled = preprocessor.fit_transform(X)

    columns = ['tempo', 'loudness', 'time_signature',
                'emotion_rank_1', 'emotion_rank_2',
                 'energy_rank_1', 'energy_rank_2',
                   'danceability',
                    'energy',
                    'acousticness',
                    'instrumentalness',
                    'valence']


    # Conversion en DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=columns)

    return X_scaled_df


############################ PREPROC LYRICS ############################

def preproc_lyrics(df):
    return "thomas"


def preproc_lyrics_bert(df):
    return "milene"
