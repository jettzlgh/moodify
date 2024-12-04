import os
import numpy as np

## Params to train on GCP #####
GCP_PROJECT = 'moodifyproject'
BUCKET_NAME = 'moodify_bucket'
DATA_BLOB_NAME = 'lyrics_with_labels.csv' #'lyrics_with_labels_50_songs.csv'

# Model params
BATCH_SIZE = 256
EMBEDDING_DIM = 32
GRU_LAYER = 16
DENSE_LAYER = 16
EPOCHS = 20
PATIENCE = 4
