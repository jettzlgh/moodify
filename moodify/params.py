import os
import numpy as np

## Params to train on GCP #####
GCP_PROJECT = 'moodifyproject'
BUCKET_NAME = 'moodify_bucket'
DATA_BLOB_NAME = 'lyrics_with_labels_50_songs.csv'

# Model params
BATCH_SIZE = 32
EMBEDDING_DIM = 64
GRU_LAYER = 64
DENSE_LAYER = 64
EPOCHS = 5
PATIENCE = 20
