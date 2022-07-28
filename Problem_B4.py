# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import csv
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential


def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    # Using "shuffle=False"
    sentences = bbc.text
    labels = bbc.category

    train_sentences, validation_sentences, train_labels, validation_labels = train_test_split(sentences,
                                                                                              labels,
                                                                                              train_size=training_portion,
                                                                                              shuffle=False
                                                                                              )

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)
    word_index = tokenizer.word_index

    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_sentences = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type,
                                    padding=padding_type)  # YOUR CODE HERE

    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
    validation_sentences = pad_sequences(validation_sequences, maxlen=max_length, truncating=trunc_type,
                                         padding=padding_type)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)

    train_labels = np.array(label_tokenizer.texts_to_sequences(train_labels))
    validation_labels = np.array(label_tokenizer.texts_to_sequences(validation_labels))

    model = Sequential([
        # YOUR CODE HERE.
        # YOUR CODE HERE. DO not change the last layer or test may fail
        keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(6, activation='softmax')
    ])

    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(train_sentences, train_labels, epochs=50,
                        validation_data=(validation_sentences, validation_labels))

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")
