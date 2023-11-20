import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Generating synthetic data
num_samples = 10000
input_sequence_length = 63
input_embedding_size = 128
target_embedding_size = 128

# Generate random input sequences of embeddings
input_sequences = np.random.rand(num_samples, input_sequence_length, input_embedding_size)
# Multiply input_sequences by a fixed number to get target_embeddings
fixed_multiplier = 15
# Multiply along axes to reduce dimensions
target_embeddings = np.mean(input_sequences * fixed_multiplier, axis=1)  # Example: using mean aggregation


# Split the data into training and validation sets
split_ratio = 0.8
split_index = int(num_samples * split_ratio)

train_input = input_sequences[:split_index]
train_target = target_embeddings[:split_index]

val_input = input_sequences[split_index:]
val_target = target_embeddings[split_index:]


# Transformer Model definition
def transformer_model(input_sequence_length, input_embedding_size, target_embedding_size):
    inputs = tf.keras.Input(shape=(input_sequence_length, input_embedding_size))
    # Multi-head self-attention layer
    attention_output = layers.MultiHeadAttention(
        num_heads=8, key_dim=input_embedding_size, dropout=0.1)(inputs, inputs)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + inputs)
    # Feedforward layer
    outputs = layers.Conv1D(filters=512, kernel_size=1, activation='relu')(attention_output)
    outputs = layers.GlobalAveragePooling1D()(outputs)
    # Output layer
    outputs = layers.Dense(target_embedding_size)(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="transformer_model")
    return model


# Compile the model
transformer = transformer_model(input_sequence_length, input_embedding_size, target_embedding_size)
transformer.compile(optimizer='adam', loss='mean_squared_error')
transformer.fit(train_input, train_target, validation_data=(val_input, val_target), epochs=10, batch_size=32)


# check performance
predictions = transformer.predict(val_input)
mse = mean_squared_error(val_target, predictions)
print(f"Mean Squared Error: {mse}")

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(val_target)

# Scatter plot of original embeddings and predicted embeddings in a 2D space
plt.figure(figsize=(8, 6))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], label='Original Embeddings')
plt.scatter(predictions[:, 0], predictions[:, 1], label='Predicted Embeddings')
plt.title('Embedding Space Visualization')
plt.legend()
plt.show()

# Calculate cosine similarity between predicted and actual embeddings
cos_sim = cosine_similarity(val_target, predictions)
print(f"Cosine Similarity: {cos_sim.mean()}")

# t-SNE visualization for actual embeddings
tsne = TSNE(n_components=2)
transformed_actual_embeddings = tsne.fit_transform(val_target)

# t-SNE visualization for predicted embeddings
transformed_predicted_embeddings = tsne.fit_transform(predictions)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(transformed_actual_embeddings[:, 0], transformed_actual_embeddings[:, 1], label='Actual Embeddings')
plt.title('t-SNE Visualization of Actual Embeddings')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(transformed_predicted_embeddings[:, 0], transformed_predicted_embeddings[:, 1], label='Predicted Embeddings')
plt.title('t-SNE Visualization of Predicted Embeddings')
plt.legend()

plt.tight_layout()
plt.show()