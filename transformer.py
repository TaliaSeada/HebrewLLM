import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.metrics import mean_squared_error

# Generating synthetic data
num_samples = 10000
input_sequence_length = 63
input_embedding_size = 128
target_embedding_size = 128

# Generate random input sequences of embeddings
min_val = 0
max_val = 1
input_sequences = np.random.uniform(min_val, max_val, size=(num_samples, input_sequence_length, input_embedding_size))

# # Multiply input_sequences by a fixed number to get target_embeddings
# fixed_multiplier = 15
# # Multiply along axes to reduce dimensions
# target_embeddings = input_sequences*fixed_multiplier

# Create a multiplier array with increasing values from 1 to the length of input_sequences
multiplier_array = np.arange(1, num_samples + 1).reshape(-1, 1, 1)
# Element-wise multiplication between input_sequences and multiplier_array
target_embeddings = input_sequences * multiplier_array
target_embeddings = target_embeddings + 13

# target_embeddings = input_sequences


# Split the data into training and validation sets
split_ratio = 0.8
split_index = int(num_samples * split_ratio)

train_input = input_sequences[:split_index]
train_target = target_embeddings[:split_index]

val_input = input_sequences[split_index:]
val_target = target_embeddings[split_index:]


# Transformer Model definition
def transformer_model():
    inputs = tf.keras.Input(shape=(input_sequence_length, input_embedding_size))
    # Multi-head self-attention layer
    attention_output = layers.MultiHeadAttention(num_heads=10, key_dim=input_embedding_size, dropout=0.1)(inputs,
                                                                                                          inputs)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + inputs)
    # Feedforward layer
    outputs = layers.Conv1D(filters=512, kernel_size=1, activation='relu')(attention_output)
    outputs = layers.GlobalAveragePooling1D()(outputs)
    outputs = layers.Dense(target_embedding_size)(outputs)

    # Regularization
    outputs = layers.Dropout(0.1)(outputs)
    # outputs = layers.BatchNormalization()(outputs)

    # Output layer
    outputs = layers.Dense(target_embedding_size)(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="transformer_model")
    return model


# Reshape target embeddings to match model output shape
train_target_reshaped = np.mean(train_target, axis=1)
val_target_reshaped = np.mean(val_target, axis=1)


# Normalizing input and output data
def normalize_data(data):
    mean = np.mean(data, axis=(0, 1))
    std = np.std(data, axis=(0, 1))
    return (data - mean) / std


train_input_norm = normalize_data(train_input)
val_input_norm = normalize_data(val_input)
train_target_norm = normalize_data(train_target_reshaped)
val_target_norm = normalize_data(val_target_reshaped)

# Compile the model
transformer = transformer_model()
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
# Retrain the model with normalized data
transformer.compile(optimizer=custom_optimizer, loss='mean_squared_error')
transformer.fit(train_input_norm, train_target_norm, validation_data=(val_input_norm, val_target_norm), epochs=10,
                batch_size=64)

# Check performance
predictions_norm = transformer.predict(val_input_norm)
mse_norm = mean_squared_error(val_target_norm, predictions_norm)
print(f"Normalized Mean Squared Error: {mse_norm}")
