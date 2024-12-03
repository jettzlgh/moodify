
from google.cloud import storage
import pandas as pd
import pickle
from moodify.params import *
import sys

from datetime import datetime
import warnings
import time

from moodify.preproc import preproc_rnn, preproc_rnn_bert, mood_filter
from moodify.model import set_model_rnn, fit_model_rnn, set_model_bert, scrolling_prediction, scrolling_prediction_bert

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
    else:
        data_blob_name = 'lyrics_with_labels.csv'
        epochs = 200

    # Record the start time
    start_time = time.time()

    # Get the raw data from the moodify bucket ###############

    gcs_path = f'gs://{BUCKET_NAME}/{data_blob_name}'
    print('Loading data from the following bucket:',gcs_path)
    df = pd.read_csv(gcs_path)

    print("✅ accessed df on bucket \n")

    # Keep only the select class
    df = mood_filter(df,cluster=class_code)

    # Preprocess data ##############################

    if model_type == 'bert':
        # start
        inputs, targets, tokenizer = preproc_rnn_bert(df,word_bucket)
        # end
        print('prerocessing for BERT model')
    elif model_type == 'rnn':
        #start
        inputs, targets, tokenizer = preproc_rnn(df, word_bucket) #
        #end
        print('preprocessing for RNN model')
    else:
        print('incompatible model')
        return ValueError

    # END
    print("✅ preprocessing done \n")
    end_time_preproc = time.time()

    # Train model ###################################

    if model_type == "bert":
        # start
        model = set_model_bert(X= inputs, y = targets, tokenizer = tokenizer,
                  word_bucket = word_bucket,
                  gru_layer = GRU_LAYER,
                  dense_layer = DENSE_LAYER)
        # end
        print(f'training BERT model for class {class_code}')

        model, history = fit_model_rnn(model, inputs, targets, epochs=epochs, patience = PATIENCE, batch_size=BATCH_SIZE)


    if model_type == "rnn":

        print(f'training RNN model for class {class_code} \n')

        model = set_model_rnn(X= inputs, y = targets, tokenizer = tokenizer,
                  word_bucket = word_bucket,
                  embedding_dim = EMBEDDING_DIM,
                  gru_layer = GRU_LAYER,
                  dense_layer = DENSE_LAYER)

        model, history = fit_model_rnn(model, inputs, targets, epochs=epochs, patience = PATIENCE, batch_size=BATCH_SIZE)

        print(f'✅ finished training RNN model \n')
    end_time_train = time.time()


    # Save model and tokenizer ####################################

    print(f'saving training RNN model for class {class_code} \n')

    # Create a unique model name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    inputs_filename = f"{model_type}_c{class_code}_wb{word_bucket}_{timestamp}_inputs.pkl"
    targets_filename = f"{model_type}_c{class_code}_wb{word_bucket}_{timestamp}_targets.pkl"
    model_filename = f"{model_type}_c{class_code}_wb{word_bucket}_{timestamp}_model.pkl"
    tokenizer_filename = f"{model_type}_c{class_code}_wb{word_bucket}_{timestamp}_tokenizer.pkl"
    history_filename = f"{model_type}_c{class_code}_wb{word_bucket}_{timestamp}_history.pkl"

    # Save a copy of the model where you are
    # NOTE: does this need to be optimized for the VC?
    local_path = os.getcwd()

    # Create the save paths
    inputs_path = os.path.join(local_path, inputs_filename)
    targets_path = os.path.join(local_path, targets_filename)
    model_path = os.path.join(local_path, model_filename)
    tokenizer_path = os.path.join(local_path, tokenizer_filename)
    history_path = os.path.join(local_path, history_filename)

    warnings.filterwarnings("ignore", category=UserWarning, module="absl")

    # Save inputs preprocessed
    with open(inputs_path, 'wb') as file:
        pickle.dump(inputs, file)

    # Save targets preprocessed
    with open(targets_path, 'wb') as file:
        pickle.dump(targets, file)

    # Save as a tensorflow model
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

    # Save the tokenizer as a pickle
    with open(tokenizer_path, 'wb') as file:
        pickle.dump(tokenizer, file)

    # Save the model history
    with open(history_filename, 'wb') as file:
        pickle.dump(history, file)


    if model_target == "gcs":

        # Initialize GCP client
        client = storage.Client()  # Access GCP

        bucket = client.bucket(BUCKET_NAME)  # Replace with your GCP bucket name
        print(f'Accessing client at: {client} \n')

        # Define the blob (path inside the bucket where the model will be stored)
        inputs_blob = bucket.blob(f"models/{inputs_filename}")
        targets_blob = bucket.blob(f"models/{targets_filename}") # Save inside the 'models' folder
        tokenizer_blob = bucket.blob(f"models/{tokenizer_filename}")
        model_blob = bucket.blob(f"models/{model_filename}")  # Save inside the 'models' folder
        tokenizer_blob = bucket.blob(f"models/{tokenizer_filename}")
        history_blob = bucket.blob(f"models/{history_filename}")

        # Upload the pickle file to the GCP bucket
        inputs_blob.upload_from_filename(inputs_path)
        targets_blob.upload_from_filename(targets_path)
        model_blob.upload_from_filename(model_path)  # Upload the file to the bucket
        tokenizer_blob.upload_from_filename(tokenizer_path)
        history_blob.upload_from_filename(history_filename)
        print('Finished uploading \n')

        # Delete the local model file after upload (remove from the VM)
        # NOTE: Is this step needed?
        os.remove(inputs_path)
        os.remove(targets_path)
        os.remove(model_path)
        os.remove(tokenizer_path)
        os.remove(history_path)
        print('Local files deleted \n')

        print(f"✅Model, tokeniser and history saved to GCP bucket \n")


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

    print(f'Time taken: {elapsed_time:.2f} (preproc: {elapsed_preproc:.2f}, train: {elapsed_train:.2f}, save: {elapsed_save:.2f}) ')


    return None

if __name__ == "__main__":
  model_type = str(sys.argv[1])
  class_code = int(sys.argv[2])
  model_target = str(sys.argv[3])
  word_bucket = int(sys.argv[4])
  run_type = str(sys.argv[5])

  model_train(model_type, class_code, model_target, word_bucket, run_type)
