# ==========================================================================================================
# PROBLEM A4
#
# Build and train a binary classifier for the IMDB review dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in http://ai.stanford.edu/~amaas/data/sentiment/
#
# Desired accuracy and validation_accuracy > 83%
# ===========================================================================================================

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras.models import Sequential


def solution_A4():
    imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
    # YOUR CODE HERE
    train_data, test_data = imdb['train'], imdb['test']
    train_sentences =[]
    test_sentences =[]
    train_labels =[]
    test_labels =[]

    # DO NOT CHANGE THIS CODE
    for s, l in train_data:
        train_sentences.append(s.numpy().decode('utf8'))
        train_labels.append(l.numpy())

    for s, l in test_data:
        test_sentences.append(s.numpy().decode('utf8'))
        test_labels.append(l.numpy())

    # YOUR CODE HERE
    len(test_sentences),test_sentences[:10],test_labels[:10]
    len(train_sentences),train_sentences[:10],train_labels[:10],
    train_labels_result = np.array(train_labels)
    test_labels_result = np.array(test_labels)
    train_labels_result[:10]

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or test may fail
    vocab_size = 10000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    oov_tok = "<OOV>"

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)
    word_index = tokenizer.word_index

    sequences = tokenizer.texts_to_sequences(train_sentences)
    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    pad = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)
    test_pad = pad_sequences(test_sequences,maxlen=max_length)# YOUR CODE HERE

    reverse = dict([(value, key) for (key, value) in word_index.items()])

    def decode_review(text):
        return ' '.join([reverse.get(a, '?') for a in text])

    model = Sequential([
        # YOUR CODE HERE. Do not change the last layer.
        keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(6, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    history = model.fit(pad, train_labels_result, epochs=5, validation_data=(test_pad, test_labels_result), verbose =1)

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_A4()
    model.save("model_A4.h5")
