import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense


def input_target_rnn(X_token, word_bucket):
    '''
    Transforms the preproc X_token into the model input and target.
    Note: at the end of the song, the last words are dropped as there is no target

    Returns: X and y for the function model_rnn, as np.arrays

    Parameters:
    - word_bucket: Length of the input sequences (max length of each sentence)

    '''
    inputs, targets = [], []
    # Take a list of lists of tokens (each one song lyrics)

    for song in X_token:
        # Convert sentence to a NumPy array for efficient slicing
        song_array = np.array(song)

        # Create the input-target pairs by shuffling
        for i in range(len(song) - word_bucket):
            inputs.append(song_array[i:i + word_bucket])
            targets.append(song_array[i + word_bucket])

    # Make sure that input and targte are NumPy arrays
    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


def model_rnn(X, y, vocab_size, embedding_dim, word_bucket):
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
    - X_token: a list of songs, each with tokenized words
    - vocab_size: Number of unique words in your vocab
    - embedding_dim:  Dimensionality of the word embeddings
    - word_bucket: Length of the input sequences (max length of each sentence)
    '''

    # Start the model
    model = Sequential()

    # Embedding layer
    model.add(Embedding(input_dim=vocab_size + 1,  #
                    output_dim=embedding_dim,  #
                    mask_zero=True,  # Mask the padding
                    input_length=word_bucket))

    # GRU layer
    model.add(GRU(64, return_sequences=False))
    # NOTE: Do i need to return sequences?

    # Dense layer (hidden layer)
    model.add(Dense(64, activation='relu'))

    # Output layer (softmax activation for multi-class classification)
    model.add(Dense(vocab_size, activation='softmax'))

    # NOTE: does the output need to be vocab_size + 1?

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy',  # For multi-class classification
              optimizer='adam',
              metrics=['accuracy'])

    print(model.summary())

    return model


def scrolling_prediction(model, tokenizer, seed_text, max_length, num_predictions=100):
    """
    Generate a sequence of words using a trained model and tokenizer.

    Returns:
    - full_generated_text: The complete predicted text as a list of words.

    Parameters:
    - model: Trained word prediction model.
    - tokenizer: Tokenizer used during training.
    - seed_text: List of initial words to start the prediction (e.g., ['i', 'was', 'in']).
    - max_length: The number of words the model expects as input (e.g., 3 for this RNN).
    - num_predictions: Number of words to predict.
    """

    # Convert seed_text to tokens
    input_sequence = tokenizer.texts_to_sequences([seed_text])[0]

    # Ensure input_sequence is the right length by padding or truncating
    if len(input_sequence) < max_length:
        input_sequence = [0] * (max_length - len(input_sequence)) + input_sequence
    else:
        input_sequence = input_sequence[-max_length:]

    # Initialize the generated text
    full_generated_text = seed_text[:]

    # Generate words iteratively
    for _ in range(num_predictions):
        # Reshape input_sequence for the model (1, max_length)
        input_array = np.array(input_sequence).reshape(1, max_length)

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
        input_sequence = input_sequence[-max_length:]  # Keep only the last max_length tokens

    return full_generated_text  # Join words into a sentence
