
from google.cloud import storage
import pandas as pd
import pickle
from moodify.params import *
import sys

from datetime import datetime

from moodify.preproc import preproc_rnn, preproc_rnn_bert, mood_filter
from moodify.model import set_model_rnn, fit_model_rnn, set_model_bert, scrolling_prediction, scrolling_prediction_bert

def model_train(model_type, class_code, model_target, word_bucket):
    """
    NOTE: ideal would be to loop over the class codes

    - Trains RNN and BERT models
    - Download data with labels from bucket
    - Preprocess (according to model type)
    - Train the model
    - Store model in model bucket
    """

    # Get the raw data from the moodify bucket ###############

    gcs_path = f'gs://{BUCKET_NAME}/{DATA_BLOB_NAME}'
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

    # Train model ###################################

    if model_type == "bert":
        # start
        model = set_model_bert(X= inputs, y = targets, tokenizer = tokenizer,
                  word_bucket = word_bucket,
                  gru_layer = GRU_LAYER,
                  dense_layer = DENSE_LAYER)
        # end
        print(f'training BERT model for class {class_code}')

        model, history = fit_model_rnn(model, inputs, targets, epochs=EPOCHS, patience = PATIENCE, batch_size=BATCH_SIZE)


    if model_type == "rnn":

        print(f'training RNN model for class {class_code}')

        model = set_model_rnn(X= inputs, y = targets, tokenizer = tokenizer,
                  word_bucket = word_bucket,
                  embedding_dim = EMBEDDING_DIM,
                  gru_layer = GRU_LAYER,
                  dense_layer = DENSE_LAYER)

        model, history = fit_model_rnn(model, inputs, targets, epochs=EPOCHS, patience = PATIENCE, batch_size=BATCH_SIZE)

        print(f'✅ finished training RNN model for class {class_code}')



    # Save model and tokenizer ####################################

    print(f'saving training RNN model for class {class_code}')

    # Create a unique model name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_filename = f"{model_type}_c{class_code}_wb{word_bucket}_{timestamp}_model.pkl"
    tokenizer_filename = f"{model_type}_c{class_code}_wb{word_bucket}_{timestamp}_tokenizer.pkl"
    history_filename = f"{model_type}_c{class_code}_wb{word_bucket}_{timestamp}_history.pkl"

    # Save a copy of the model where you are
    # NOTE: does this need to be optimized for the VC?
    local_path = os.getcwd()

    # Create the save paths
    model_path = os.path.join(local_path, model_filename)
    tokenizer_path = os.path.join(local_path, tokenizer_filename)
    history_path = os.path.join(local_path, history_filename)

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
        print('Accessing client at:',client)

        # Define the blob (path inside the bucket where the model will be stored)
        model_blob = bucket.blob(f"models/{model_filename}")  # Save inside the 'models' folder
        tokenizer_blob = bucket.blob(f"models/{tokenizer_filename}")
        history_blob = bucket.blob(f"models/{history_filename}")

        # Upload the pickle file to the GCP bucket
        model_blob.upload_from_filename(model_path)  # Upload the file to the bucket
        tokenizer_blob.upload_from_filename(tokenizer_path)
        history_blob.upload_from_filename(history_filename)
        print('Finished uploading')

        # Delete the local model file after upload (remove from the VM)
        # NOTE: Is this step needed?
        os.remove(model_path)
        os.remove(tokenizer_path)
        os.remove(history_path)
        print('Local files deleted')

        print(f"✅Model, tokeniser and history saved to GCP bucket")


    if model_target == "local":

        print(f"✅ Model saved locally as '{model_filename}'")
        print(f"✅ Tokenizer saved as '{tokenizer_filename}'")

    return None

if __name__ == "__main__":
  model_type = sys.argv[1]
  class_code = sys.argv[2]
  model_target = sys.argv[3]
  word_bucket = sys.argv[4]

  model_train(model_type, class_code, model_target, word_bucket)
