from google.cloud import storage
import pandas as pd
import pickle
from moodify.params import *

from datetime import datetime

from moodify.preproc import preproc_rnn, preproc_rnn_bert, mood_filter
from moodify.model import set_model_rnn, fit_model_rnn

def model_train(model_type, class_code, model_target):
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

    # Preprocess data ##############################

    if model_type == 'bert':
        # start

        # end
        print('prerocessing for BERT model')
    elif model_type == 'rnn':
        #start
        df = mood_filter(df,cluster=class_code) # Keep only the select class
        inputs, targets, tokenizer = preproc_rnn(df, WORD_BUCKET) #
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

        # end
        print(f'training BERT model for class {class_code}')

    if model_type == "rnn":

        model = set_model_rnn(X= inputs, y = targets, tokenizer = tokenizer,
                  word_bucket = WORD_BUCKET,
                  embedding_dim = EMBEDDING_DIM,
                  gru_layer = GRU_LAYER,
                  dense_layer = DENSE_LAYER)

        model, history = fit_model_rnn(model, inputs, targets, epochs=EPOCHS, patience = PATIENCE, batch_size=BATCH_SIZE)

        print(f'training RNN model for class {class_code}')


    # Save model and tokenizer ####################################

    # Create a unique model name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_type}_{class_code}_{timestamp}_model.pkl"
    tokenizer_filename = f"{model_type}_{class_code}_{timestamp}_tokenizer.pkl"

    # Save a copy of the model where you are
    # NOTE: does this need to be optimized for the VC?
    local_path = os.getcwd()

    # Create the save paths
    model_path = os.path.join(local_path, model_filename)
    tokenizer_path = os.path.join(local_path, tokenizer_filename)

    # Save as a tensorflow model
    model.save(model_path, save_format='tf')

    # Save the tokenizer as a pickle
    with open(tokenizer_path, 'wb') as file:
        pickle.dump(tokenizer, file)  # Assuming `model` is your trained model object

    if model_target == "gcs":
        
        # Initialize GCP client
        client = storage.Client()  # Access GCP
        bucket = client.bucket(BUCKET_NAME)  # Replace with your GCP bucket name

        # Define the blob (path inside the bucket where the model will be stored)
        model_blob = bucket.blob(f"models/{model_filename}")  # Save inside the 'models' folder
        tokenizer_blob = bucket.blob(f"models/{tokenizer_filename}")

        # Upload the pickle file to the GCP bucket
        model_blob.upload_from_filename(model_path)  # Upload the file to the bucket
        tokenizer_blob.upload_from_filename(tokenizer_path)

        # Delete the local model file after upload (remove from the VM)
        # NOTE: Is this step needed?
        os.remove(model_path)
        os.remove(tokenizer_path)

        print(f"✅Model and tokeniser saved to GCP bucket")


    if model_target == "local":

        print(f"✅ Model saved locally as '{model_filename}'")
        print(f"✅ Tokenizer saved as '{tokenizer_filename}'")

    return None
