"""a transformer that acts as a converter between the output of the llama2 network and the input of the translation
network """

import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers

# import the data in order to get the size of the embedded output
# ["cities", "inventions", "elements", "animals", "facts", "companies", "generated"]
list_of_datasets = ["generated"]

# TODO check the tranformer
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


for dataset_to_use in list_of_datasets:
    file = "output\\" + "embeddings_with_labels_" + dataset_to_use + "350m_1_rmv_period.csv"
    df_input = pd.read_csv(file)
    # Convert the string representation of embeddings to actual lists
    df_input['embeddings'] = df_input['embeddings'].apply(eval)
    # Find the longest sentence
    max_length_input = max(df_input['embeddings'].apply(lambda x: len(max(x, key=len))))
    # Pad the other sentences to the max length
    df_input['embeddings'] = df_input['embeddings'].apply(
        lambda x: x + [[0] * len(x[0])] * (max_length_input - len(x)))
    # min_length_input = min(df_input[embedding_column].apply(lambda x: len(min(x, key=len))))  # check

    statement_column = 'statement'
    # Find the longest sentence
    max_length_statement = max(df_input[statement_column].apply(len))
    # Pad the other sentences to the max length
    df_input[statement_column] = df_input[statement_column].apply(lambda x: x + ' ' * (max_length_statement - len(x)))
    # min_length_statement = min(df_input[statement_column].apply(len)) #check

    file = "output_trans\\" + "translate_with_labels_" + dataset_to_use + "_1_rmv_period.csv"
    df_output = pd.read_csv(file)
    # Convert the string representation of embeddings to actual lists
    df_output['embeddings'] = df_output['embeddings'].apply(eval)
    # Find the longest innermost sentence
    max_length_output = max(df_output['embeddings'].apply(lambda x: len(max(x[0], key=len))))
    # Pad the other sentences to the max length
    df_output['embeddings'] = df_output['embeddings'].apply(lambda x: [x[0] + [0] * (max_length_output - len(x[0]))])
    # min_length_output = min(df_output[embedding_column].apply(lambda x: len(x[0])))  # check

    # TODO make train set and test set
    train_input_embeddings = df_input['embeddings']
    train_output_embeddings = df_output['embeddings']

    # Create the transformer model
    model = transformer_model(max_length_statement, max_length_input, max_length_output)
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model
    model.fit(train_input_embeddings, train_output_embeddings, epochs=50, batch_size=32, validation_split=0.2)
    # Example of using the trained model for prediction
    example_input_embedding = np.random.rand(1, max_length_statement, max_length_input)
    predicted_target_embedding = model.predict(example_input_embedding)
