import pandas as pd
import numpy as np

# Modelling
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense

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


# def fit_model_rnn(model, X, mood):
#     model.fit(X)
#     return



def scrolling_prediction(model, tokenizer, seed_text, word_bucket, num_predictions=100):
    """
    Generate a sequence of words using a trained model and tokenizer.

    Returns:
    - full_generated_text: The complete predicted text as a list of words.

    Parameters:
    - model: Trained word prediction model.
    - tokenizer: Tokenizer used during training.
    - seed_text: List of initial words to start the prediction (e.g., ['i', 'was', 'in']).
    - word_bucket: The number of words the model expects as input (e.g., 3 for this RNN).
    - num_predictions: Number of words to predict.
    """

    # Convert seed_text to tokens
    input_sequence = tokenizer.texts_to_sequences([seed_text])[0]

    # Ensure input_sequence is the right length by padding or truncating
    if len(input_sequence) < word_bucket:
        input_sequence = [0] * (word_bucket - len(input_sequence)) + input_sequence
    else:
        input_sequence = input_sequence[-word_bucket:]

    # Initialize the generated text
    full_generated_text = seed_text[:]

    # Generate words iteratively
    for _ in range(num_predictions):
        # Reshape input_sequence for the model (1, word_bucket)
        input_array = np.array(input_sequence).reshape(1, word_bucket)

        # Predict the next word (output probabilities)
        predictions = model.predict(input_array, verbose=0)

        # Choose the word with the highest probability
        predicted_word_index = np.argmax(predictions, axis=-1)[0]

        # Convert the predicted index back to a word
        predicted_word = tokenizer.index_word.get(predicted_word_index, '')

        # Stop if no valid prediction
        if not predicted_word:
            break

        # Append the predicted word to the full text
        full_generated_text += " " + predicted_word

        # Update the input_sequence to include the new word
        input_sequence.append(predicted_word_index)
        input_sequence = input_sequence[-word_bucket:]  # Keep only the last max_length tokens

    return full_generated_text  # Join words into a sentence
