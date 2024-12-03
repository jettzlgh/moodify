import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
#from accelerate import disk_offload , init_empty_weights, infer_auto_device_map


def get_transformers_lyrics():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    # with init_empty_weights():
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # device_map = infer_auto_device_map(
    #     model,
    #     max_memory={"cpu": "16GB", "gpu": "8GB", "disk": "100GB"}  # Adjust limits to your setup
    # )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    # disk_offload(model=model, offload_dir="offload")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


# Modelling
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from moodify.preproc import preproc_rnn_bert

# Tokenizers
from tensorflow.keras.preprocessing.text import Tokenizer

def set_model_rnn(X, y, tokenizer, word_bucket, embedding_dim = 32, gru_layer =64, dense_layer=64):
    '''
    Returns:
        - A compiled RNN model ready to be fitted.

    Starts and compiles a sequential RNN model with:
    - An embedding layer (based on the word_bucket size)
    - A GRU layer
    - A Dense layer (relu)
    - A Dense output layer (softmax, for word probability)
    - with an adam optimizer

    Parameters:
    - X: inputs, tokenizes
    - y: model targets
    - tokenizer: tokenizer used
    - word_bucket: Length of the input sequences (max length of each sentence)
    - embedding_dim:  Dimensionality of the word embeddings
    - gru_layer: number of GRU neurons
    - dense_layer: number of neurons in the dense layers
    '''
    # Get the vocab size
    vocab_size = len(tokenizer.word_index) + 1

    # Start the model
    model = Sequential()

    # Embedding layer
    model.add(Embedding(input_dim=vocab_size + 1,  #
                    output_dim=embedding_dim,  #
                    mask_zero=True,  # Mask the padding
                    input_length=word_bucket))

    # GRU layer
    model.add(GRU(gru_layer, return_sequences=False))

    # Dense layer (hidden layer)
    model.add(Dense(dense_layer, activation='relu'))

    # Output layer
    model.add(Dense(vocab_size, activation='softmax'))

    # NOTE: does the output need to be vocab_size + 1?

    # Compile the model

    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    total_params = sum([tf.reduce_prod(tf.shape(w)) for layer in model.layers for w in layer.get_weights()])

    print(f"Total number of trainable parameters: {total_params}")

    return model


def set_model_bert(X,y,tokenizer,word_bucket,gru_layer =64,dense_layer=64):
    '''
    Returns:
        - A compiled RNN model ready to be fitted with BERT spec.

    Starts and compiles a sequential RNN model adapted to BERT tokenizer with:
    - A Dense layer (relu)
    - A Dense output layer (softmax, for word probability)
    - with an adam optimizer

    Parameters:
    - X: inputs, tokenizes
    - y: model targets
    - tokenizer: tokenizer used
    - word_bucket: Length of the input sequences (max length of each sentence)
    - embedding_dim:  Dimensionality of the word embeddings
    - gru_layer: number of GRU neurons
    - dense_layer: number of neurons in the dense layers
    '''

    # Start the model
    model = Sequential()

    # GRU layer
    model.add(GRU(gru_layer, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))

    # Dense layer (hidden layer)
    model.add(Dense(dense_layer, activation='relu'))

    # Output layer
    vocab_size = y.shape[-1]  # Number of classes (vocab size from BERT)
    model.add(Dense(vocab_size, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', # loss w/o BERT : 'sparse_categorical_crossentropy'
              optimizer='adam',
              metrics=['accuracy'])

    total_params = sum([tf.reduce_prod(tf.shape(w)) for layer in model.layers for w in layer.get_weights()])

    print(f"Total number of trainable parameters: {total_params}")

    return model


def fit_model_rnn(model, X, y, epochs=10, patience = 10, batch_size=32):
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    # Fit the model with early stopping and model checkpoint
    history = model.fit(np.array(X), np.array(y), epochs=epochs,
                validation_split=0.3,
                batch_size = batch_size,
                shuffle = True,
                callbacks=[early_stopping])
    return model, history


def fit_model_rnn_with_checkpoint(model, X, y, epochs=10, patience = 10, batch_size=32, save_path="model.h5"):
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    checkpoint = ModelCheckpoint(save_path, save_best_only=True, monitor='val_loss', mode='min')

    # Fit the model with early stopping and model checkpoint
    history = model.fit(X, y, epochs=epochs,
                validation_split=0.3,
                batch_size = batch_size,
                shuffle = True,
                callbacks=[checkpoint, early_stopping])
    return model, history


def scrolling_prediction(model, tokenizer, seed_text, word_bucket, num_predictions=100):
    """
    Generate a sequence of words using a trained model and tokenizer.

    Returns:
    - full_generated_text: The complete predicted text as a string.

    Parameters:
    - model: Trained word prediction model.
    - tokenizer: Tokenizer used during training.
    - seed_text: List of initial words to start the prediction (e.g., ['i', 'was', 'in']).
    - word_bucket: The number of words the model expects as input (e.g., 3 for this RNN).
    - num_predictions: Number of words to predict.
    """
    # Convert seed_text to tokens
    seed_token = tokenizer.texts_to_sequences([seed_text])[0]  # Convert to token list

    # Ensure the seed_token length matches the word_bucket size
    if len(seed_token) < word_bucket:
        # Pad with zeros at the beginning if too short
        seed_token = [0] * (word_bucket - len(seed_token)) + seed_token
    else:
        # Truncate to the most recent `word_bucket` tokens if too long
        seed_token = seed_token[-word_bucket:]

    # Initialize generated text and tokens
    generated_text = seed_text
    generated_tokens = np.array(seed_token).reshape(1, word_bucket)  # Shape: (1, word_bucket)

    # Generate the predicted words for the desired length
    for _ in range(num_predictions):
        # Use the last `word_bucket` tokens as input
        input_tokens = generated_tokens[:, -word_bucket:]  # Shape: (1, word_bucket)

        # Predict the next word probabilities
        prediction = model.predict(input_tokens, verbose=0)

        # Choose the word with the highest probability
        predicted_word_index = np.argmax(prediction, axis=-1)[0]

        # Convert the predicted index back to a word
        predicted_word = tokenizer.index_word.get(predicted_word_index, '')

        # If no valid word is found, skip the prediction
        if not predicted_word:
            print("Unknown word index:", predicted_word_index)
            break

        # Append the predicted word to the full text
        generated_text += " " + predicted_word

        # Update the input tokens to include the new word
        predicted_word_array = np.array([[predicted_word_index]])  # Shape: (1, 1)
        generated_tokens = np.append(generated_tokens, predicted_word_array, axis=1)

    return generated_text


def scrolling_prediction_bert(model, tokenizer, seed_text, word_bucket, hidden_dim=128, num_predictions=100):
    """
    Generate a sequence of words using a BERT-based encoder and a GRU-based decoder.

    Returns:
    - full_generated_text: The complete predicted text as a string.

    Parameters:
    - model: Pretrained BERT encoder model (e.g., TFAutoModel).
    - tokenizer: Tokenizer used with the BERT model.
    - seed_text: Initial text to start the prediction (e.g., 'i was in').
    - word_bucket: The number of tokens the decoder expects as input.
    - hidden_dim: Dimensionality of the hidden state in the GRU decoder.
    - num_predictions: Number of words to predict.
    """
    # Tokenize the seed text
    input_tokens = preproc_rnn_bert(seed_text, word_bucket=int(word_bucket))[0]
    # input_tokens = tokenizer(seed_text, return_tensors='tf', padding=True, truncation=True, max_length=word_bucket)

    # Initialize generated text
    generated_text = seed_text

    # Define the decoder (GRU + Dense layers)
    gru_decoder = tf.keras.layers.GRU(hidden_dim, return_sequences=True, return_state=True)
    dense_decoder = tf.keras.layers.Dense(tokenizer.vocab_size, activation='softmax')

    # Initialize the hidden state for the GRU
    decoder_hidden_state = tf.zeros((1, hidden_dim))  # Batch size = 1, hidden_dim
    # print("coucou", input_tokens)

    for _ in range(num_predictions):
        if _ == 0:
            input_tokens = preproc_rnn_bert(seed_text, word_bucket=int(word_bucket))[0]
        else:
            input_tokens = preproc_rnn_bert(input_tokens, word_bucket=int(word_bucket))[0]
        # Get the embeddings from the BERT encoder
        outputs = model(input_tokens) # pas de "input_ids" pour seed_text
        # print(outputs)
        last_hidden_state = outputs  # Shape: (1, seq_len, hidden_dim)
        #print("last_hidden_state shape:", last_hidden_state.shape)
        #print(last_hidden_state)
        #print("coucou", outputs)

        # Use the last token's embedding to feed into the GRU
        decoder_input = tf.expand_dims(last_hidden_state, axis=1)  # Reshape to (batch_size, 1, hidden_dim)

        # Adjust hidden state dynamically to match the batch size of decoder_input
        batch_size = tf.shape(decoder_input)[0]  # Dynamically fetch the batch size
        decoder_hidden_state = tf.zeros((batch_size, hidden_dim))  # Initialize hidden state with correct batch size

        # Pass through the GRU decoder
        gru_output, decoder_hidden_state = gru_decoder(decoder_input, initial_state=decoder_hidden_state)

        # Generate the next word prediction
        predictions = dense_decoder(gru_output[:, -1, :])  # Shape: (1, vocab_size)

        # Get the most probable token
        predicted_word_index = tf.argmax(predictions, axis=-1).numpy()[0]

        # Convert the token back to a word
        predicted_word = tokenizer.decode([predicted_word_index], skip_special_tokens=True).strip()

        # If no valid word is found, stop the prediction
        if not predicted_word:
            print("Unknown word index:", predicted_word_index)
            break

        # Append the predicted word to the generated text
        generated_text += " " + predicted_word
        #print(type(generated_text))

        # Update input tokens to include the new word
        # print(input_tokens[0][0])
        #new_input_text = tokenizer.decode(input_tokens[0][0], skip_special_tokens=True) + " " + predicted_word
        #input_tokens = tokenizer(generated_text.split(' ')[:-3], return_tensors='tf', padding=True, truncation=True, max_length=word_bucket)['input_ids']
        input_tokens = generated_text
        # print(input_tokens)

    return generated_text
