import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.metrics import mean_squared_error

# Transformer Model definition
def transformer_model(input_embedding_size, target_embedding_size):
    inputs = tf.keras.Input(shape=(input_embedding_size,))

    # Reshape the input to make it compatible with MultiHeadAttention
    reshaped_inputs = layers.Reshape((1, input_embedding_size))(inputs)

    # Multi-head self-attention layer
    attention_output = layers.MultiHeadAttention(num_heads=10, key_dim=input_embedding_size, dropout=0.1)(
        reshaped_inputs, reshaped_inputs)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + reshaped_inputs)

    # Feedforward layer
    outputs = layers.Conv1D(filters=512, kernel_size=1, activation='relu')(attention_output)
    outputs = layers.GlobalAveragePooling1D()(outputs)

    # Regularization
    outputs = layers.Dropout(0.1)(outputs)

    # Output layer
    outputs = layers.Dense(target_embedding_size)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="transformer_model")
    return model

# # Transformer Model definition (including sequence length)
# def transformer_model(input_sequence_length, input_embedding_size, target_embedding_size):
#     inputs = tf.keras.Input(shape=(input_sequence_length, input_embedding_size))
#
#     # Multi-head self-attention layer
#     attention_output = layers.MultiHeadAttention(num_heads=10, key_dim=input_embedding_size, dropout=0.1)(inputs, inputs)
#     attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + inputs)
#     # Feedforward layer
#     outputs = layers.Conv1D(filters=512, kernel_size=1, activation='relu')(attention_output)
#     outputs = layers.GlobalAveragePooling1D()(outputs)
#     # outputs = layers.Dense(target_embedding_size)(outputs)
#
#     # Regularization
#     outputs = layers.Dropout(0.1)(outputs)
#     # outputs = layers.BatchNormalization()(outputs)
#
#     # Output layer
#     outputs = layers.Dense(target_embedding_size)(outputs)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs, name="transformer_model")
#     return model

