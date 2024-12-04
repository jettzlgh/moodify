
from google.cloud import storage
import pandas as pd
import pickle
from moodify.params import *
import sys

from datetime import datetime
import warnings
import time

from moodify.preproc import preproc_rnn, preproc_rnn_bert, mood_filter
from moodify.model import set_model_rnn, fit_model_rnn, fit_model_rnn_with_checkpoint, set_model_bert, scrolling_prediction, scrolling_prediction_bert

def model_train(model_type, class_code, model_target, word_bucket, run_type):
    """
    NOTE: ideal would be to loop over the class codes

    - Trains RNN and BERT models
    - Download data with labels from bucket
    - Preprocess (according to model type)
    - Train the model
    - Store model in model bucket
    """
    if run_type == 'test':
        data_blob_name = 'lyrics_with_labels_50_songs.csv'
        epochs = 5
        patience = 2
    else:
        data_blob_name = 'lyrics_with_labels_val.csv'
        epochs = 10
        patience = 4

    # Record the start time
    start_time = time.time()


    #1.  Check if the preproc file is available on GCS

    # Compile file names
    inputs_filename = f"{model_type}_c{class_code}_wb{word_bucket}_{timestamp}_inputs.npy"
    targets_filename = f"{model_type}_c{class_code}_wb{word_bucket}_{timestamp}_targets.npy"
    tokenizer_filename = f"{model_type}_c{class_code}_wb{word_bucket}_{timestamp}_tokenizer.pkl"

    # try:
    #     gcs_path = f'gs://{BUCKET_NAME}/{data_blob_name}'
    #     print('Loading data from the following bucket:',gcs_path)
    #     df = pd.read_csv(gcs_path)

    # else:

    # Get the raw data from the moodify bucket ###############

    gcs_path = f'gs://{BUCKET_NAME}/{data_blob_name}'
    print('Loading data from the following bucket:',gcs_path)
    df = pd.read_csv(gcs_path)

    print("✅ accessed df on bucket \n")

    # Keep only the select class
    df = mood_filter(df,cluster=class_code)

    # Preprocess data ##############################


    if model_type == 'bert':


        inputs, targets, tokenizer = preproc_rnn_bert(df,word_bucket)

        print('prerocessing for BERT model')

    elif model_type == 'rnn':
        inputs, targets, tokenizer = preproc_rnn(df, word_bucket) #

        print('preprocessing for RNN model')
    else:
        print('incompatible model')
        return ValueError

    print("✅ preprocessing done \n")

    # Save the preprocessed outputs

    # Create a unique model name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    # Check you have the correct relative paths
    local_path = os.getcwd()
    inputs_path = os.path.join(local_path, inputs_filename)
    targets_path = os.path.join(local_path, targets_filename)
    tokenizer_path = os.path.join(local_path, tokenizer_filename)

    # Save locally
    np.save(inputs_path, inputs)# inputs
    np.save(targets_path, targets)# inputs
    with open(tokenizer_path, 'wb') as file: # tokenizer
        pickle.dump(tokenizer, file)

    if model_target == "gcs":

        # Initialize GCP client
        client = storage.Client()  # Access GCP
        bucket = client.bucket(BUCKET_NAME)  # Replace with your GCP bucket name
        print(f'Accessing GCS client at: {client} \n')

        # Define the blob (path inside the bucket where the model will be stored)
        inputs_blob = bucket.blob(f"preproc/{inputs_filename}")
        targets_blob = bucket.blob(f"preproc/{targets_filename}") # Save inside the 'models' folder
        tokenizer_blob = bucket.blob(f"models/{tokenizer_filename}")

        # Upload the pickle file to the GCP bucket
        inputs_blob.upload_from_filename(inputs_path)
        targets_blob.upload_from_filename(targets_path)
        tokenizer_blob.upload_from_filename(tokenizer_path)
        print('Finished uploading to GCS\n')

        # Delete the local model file after upload (remove from the VM)
        os.remove(inputs_path)
        os.remove(targets_path)
        os.remove(tokenizer_path)
        print('Deleted local cache\n')
        print("✅ preprocessed data saved to gcp \n")

    end_time_preproc = time.time()

    # Train model ###################################

    # assign a checkpoint path
    checkpoint_path = f"checkpoint_{model_type}_c{class_code}_wb{word_bucket}_{timestamp}_model.h5"

    if model_type == "bert":

        model = set_model_bert(X= inputs, y = targets, tokenizer = tokenizer,
                  word_bucket = word_bucket,
                  gru_layer = GRU_LAYER,
                  dense_layer = DENSE_LAYER)

        print(f'started training BERT model for class {class_code}')

        model, history = fit_model_rnn_with_checkpoint(model,
                                                       inputs,
                                                       targets,
                                                       epochs=epochs,
                                                       patience = patience,
                                                       batch_size=BATCH_SIZE,
                                                       save_path=checkpoint_path)

    if model_type == "rnn":

        print(f'started training RNN model for class {class_code} \n')

        model = set_model_rnn(X= inputs, y = targets, tokenizer = tokenizer,
                  word_bucket = word_bucket,
                  embedding_dim = EMBEDDING_DIM,
                  gru_layer = GRU_LAYER,
                  dense_layer = DENSE_LAYER)

        model, history = fit_model_rnn_with_checkpoint(model,
                                                       inputs,
                                                       targets,
                                                       epochs=epochs,
                                                       patience = patience,
                                                       batch_size=BATCH_SIZE,
                                                       save_path=checkpoint_path)

        print(f'✅ finished training RNN model \n')
    end_time_train = time.time()


    # Save model  ####################################

    print(f'saving training RNN model for class {class_code} \n')

    model_filename = f"{model_type}_c{class_code}_wb{word_bucket}_{timestamp}_model.h5"
    history_filename = f"{model_type}_c{class_code}_wb{word_bucket}_{timestamp}_history.pkl"

    # Create the save paths
    model_path = os.path.join(local_path, model_filename)
    history_path = os.path.join(local_path, history_filename)

    # Supress errors that might stop the code
    warnings.filterwarnings("ignore", category=UserWarning, module="absl")

    # Save the local files
    model.save(model_path) # Save in .h5 format
    with open(history_filename, 'wb') as file: # Save history as a pickle
        pickle.dump(history, file)

    if model_target == "gcs":

        # Initialize GCP client
        client = storage.Client()  # Access GCP

        bucket = client.bucket(BUCKET_NAME)  # Replace with your GCP bucket name
        print(f'Saving model at: {client} \n')

        # Define the blob (path inside the bucket where the model will be stored)
        model_blob = bucket.blob(f"models/{model_filename}")  # Save inside the 'models' folder
        history_blob = bucket.blob(f"models/{history_filename}")

        # Upload the pickle file to the GCP bucket
        model_blob.upload_from_filename(model_path)  # Upload the file to the bucket
        history_blob.upload_from_filename(history_filename)
        print('Finished uploading \n')

        # Delete the local model file after upload (remove from the VM)
        # NOTE: Is this step needed?
        os.remove(model_path)
        os.remove(history_path)
        print('Local files deleted \n')

        print(f"✅Model and history saved to GCP bucket \n")


    if model_target == "local":

        print(f"✅ Files saved locally ")

    end_time_save = time.time()

    print('✅ Run complete: ')
    print(f'model: {model_type}')
    print(f'version: {timestamp}')
    print(f'class : {class_code}')
    print(f'word bucket: {word_bucket} \n')

    elapsed_time = end_time_save - start_time
    elapsed_preproc = end_time_preproc - start_time
    elapsed_train = end_time_train - end_time_preproc
    elapsed_save = end_time_save - end_time_train

    print(f'Time taken: {elapsed_time:.2f} \n preproc: {elapsed_preproc:.2f} \n train: {elapsed_train:.2f} \n save: {elapsed_save:.2f}) ')


    return None

if __name__ == "__main__":
  model_type = str(sys.argv[1])
  class_code = int(sys.argv[2])
  model_target = str(sys.argv[3])
  word_bucket = int(sys.argv[4])
  run_type = str(sys.argv[5])

  model_train(model_type, class_code, model_target, word_bucket, run_type)
