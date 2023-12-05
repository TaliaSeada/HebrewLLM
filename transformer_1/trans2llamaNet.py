"""a transformer that acts as a converter between the output of the translation network and the input of the llama2 network """

import pandas as pd
import numpy as np
import tensorflow as tf
from transformer_1 import transformer1
from sklearn.metrics import mean_squared_error


def flatten_vector(row):
    return np.array(row[0])


def set_data(dataset_to_use):
    # TODO get the max size, add 0 if needed (both to the input data and the target data)
    # max_size_input = 512
    # max_size_target = 512
    file = "C:\\Users\\talia\\PycharmProjects\\TranslatorGPT\\output\\" + "embeddings_" + dataset_to_use + "350m_1_heb_rmv_period.csv"
    df_target_em = pd.read_csv(file)
    # Convert the string representation of embeddings to actual lists
    df_target_em['embeddings'] = df_target_em['embeddings'].apply(eval)

    statement_column = 'statement'
    # Find the longest sentence
    max_length_statement = max(df_target_em[statement_column].apply(len))
    # Pad the other sentences to the max length
    df_target_em[statement_column] = df_target_em[statement_column].apply(
        lambda x: x + ' ' * (max_length_statement - len(x)))
    # min_length_statement = min(df_input[statement_column].apply(len)) #check
    df_target_em['embeddings'] = df_target_em['embeddings'].apply(flatten_vector)
    max_size_input = max(df_target_em['embeddings'].apply(len))

    file = "C:\\Users\\talia\\PycharmProjects\\TranslatorGPT\\output_trans\\" + "translate_" + dataset_to_use + "_1_heb_rmv_period.csv"
    df_input = pd.read_csv(file)
    # Convert the string representation of embeddings to actual lists
    df_input['embeddings'] = df_input['embeddings'].apply(eval)
    df_input['embeddings'] = df_input['embeddings'].apply(flatten_vector)
    max_size_target = max(df_input['embeddings'].apply(len))

    df_target_em = df_target_em["embeddings"]
    df_input_em = df_input["embeddings"]

    # Explode the vectors into separate rows
    df_target_em = df_target_em.apply(lambda x: pd.Series(x))
    df_input_em = df_input_em.apply(lambda x: pd.Series(x))

    return df_input_em, df_target_em, max_length_statement, max_size_input, max_size_target


# import the data in order to get the size of the embedded output
# ["cities", "inventions", "elements", "animals", "facts", "companies", "generated"]
list_of_datasets = ["cities"]


def train_model1():
    for dataset_to_use in list_of_datasets:
        df_target, df_input, statement_len, max_size_input, max_size_target = set_data(dataset_to_use)
        # Split the data into training and validation sets
        split_ratio = 0.8
        data_len = (len(df_input))
        split_index = int(data_len * split_ratio)

        train_input = df_input[:split_index]
        train_target = df_target[:split_index]

        val_input = df_input[split_index:]
        val_target = df_target[split_index:]

        # Normalize data
        # train_input_norm = transformer.normalize_data(train_input)
        # val_input_norm = transformer.normalize_data(val_input)
        # train_target_norm = transformer.normalize_data(train_target_reshaped)
        # val_target_norm = transformer.normalize_data(val_target_reshaped)

        # Create the transformer model
        model = transformer1.transformer_model(max_size_input, max_size_target)
        # Compile the model
        custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        # Retrain the model with normalized data
        model.compile(optimizer=custom_optimizer, loss='mean_squared_error')
        # Train the model
        model.fit(train_input, train_target, validation_data=(val_input, val_target), epochs=11, batch_size=32)

        # Check performance
        predictions = model.predict(val_input)
        mse = mean_squared_error(val_target, predictions)
        print(f"Mean Squared Error: {mse}")

        # Save the model weights
        model.save_weights('transformer1_model_weights.h5')

        # predictions_norm = model.predict(val_input_norm)
        # mse_norm = mean_squared_error(val_target_norm, predictions_norm)
        # print(f"Normalized Mean Squared Error: {mse_norm}")


def reload(max_size_input, max_size_target):
    # Rebuild the model architecture
    model = transformer1.transformer_model(max_size_input, max_size_target)
    custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=custom_optimizer, loss='mean_squared_error')
    model.load_weights('C:\\Users\\talia\\PycharmProjects\\TranslatorGPT\\transformer_1\\transformer1_model_weights.h5')
    return model

# if __name__ == '__main__':
#     train_model1()