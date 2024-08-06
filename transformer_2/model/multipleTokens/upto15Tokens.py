"""
This file is implementing the second transformer with more than one token
TODO use opt2 & embeddingsToTranslator as reference
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, OPTForCausalLM, AutoModelForSeq2SeqLM, MarianTokenizer, MarianMTModel
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformer_2.model.multipleTokens.embeddToTrans15Tokens import translator_activation_different_layer
import torch.nn.functional as F
import optuna
import math

# Check if CUDA is available and set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# opt
model_to_use = "350m"
opt_model = OPTForCausalLM.from_pretrained("facebook/opt-" + model_to_use)
opt_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-" + model_to_use, padding_side='left')
opt_layer = -1
opt_model.to(device)

# translator
model_name = "Helsinki-NLP/opus-mt-en-he"
translator_tokenizer = MarianTokenizer.from_pretrained(model_name)
translator_model = MarianMTModel.from_pretrained(model_name)
translator_layer = 1
translator_model.to(device)

max_length = 5


class HiddenStateTransformer(nn.Module):
    def __init__(self, input_size, output_size, num_layers, num_heads, dim_feedforward, dropout=0.1):
        super(HiddenStateTransformer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Transformer Encoder Layer
        encoder_layers = TransformerEncoderLayer(d_model=input_size, nhead=num_heads,
                                                 dim_feedforward=dim_feedforward, dropout=dropout, activation=F.relu)
        encoder_layers.self_attn.batch_first = True
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers,
                                                      enable_nested_tensor=1 - (num_heads % 2))

        # Dropout layer after transformer encoder
        self.dropout = nn.Dropout(p=dropout)

        # Linear layer to map to the target hidden size
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, output_size)
        )

    def forward(self, src):
        # src shape: (seq_length, batch_size, input_size)
        encoded = self.transformer_encoder(src)
        # Apply dropout after transformer encoder
        encoded = self.dropout(encoded)
        # encoded shape: (seq_length, batch_size, input_size)
        output = self.fc(encoded)
        # output shape: (seq_length, batch_size, output_size)
        return output


# Custom Dataset to handle pairs of tensors
class TensorDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Custom collate function
def custom_collate_fn(batch):
    X_batch, Y_batch = zip(*batch)
    X_padded = pad_sequence([x.squeeze(0) for x in X_batch], batch_first=True, padding_value=0)
    Y_padded = pad_sequence([y.squeeze(0) for y in Y_batch], batch_first=True, padding_value=0)
    return X_padded, Y_padded


# Label smoothing function
def add_noise_to_targets(targets, noise_factor=0.1):
    noise = torch.randn_like(targets) * noise_factor
    return targets + noise * (1 - noise_factor) + noise_factor / targets.size(-1)


# Training function
def train_transformer(train_loader, val_loader, input_size, output_size, num_layers, num_heads, dim_feedforward,
                      dropout, epochs=10, learning_rate=0.01, noise_factor=0.1):
    model = HiddenStateTransformer(input_size, output_size, num_layers, num_heads, dim_feedforward, dropout)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_loader),
                                                    epochs=epochs)
    model.to(device)

    best_val_loss = float('inf')
    early_stopping_patience = 3
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            # Apply label smoothing by adding noise to the targets
            Y_batch_smoothed = add_noise_to_targets(Y_batch, noise_factor)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch_smoothed)
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

    return model, val_loss


# Evaluation function
def evaluate_model(model, test_loader, criterion, num_prints=5):
    model.eval()
    total_loss = 0
    print_count = 0

    with torch.no_grad():
        for batch_idx, (X_batch, Y_batch) in enumerate(test_loader):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            total_loss += loss.item()

            # Print some of the results
            if print_count < num_prints:
                # Get the original sentences from the dataset
                input_sentence = df.iloc[batch_idx]['English sentence']
                expected_sentence = df.iloc[batch_idx]['Hebrew sentence']

                # Decode the output sequence for comparison
                output_sentence = \
                    translator_tokenizer.batch_decode(outputs[0].argmax(dim=-1).unsqueeze(0), skip_special_tokens=True)[
                        0]

                print(f"Input Sentence {print_count + 1}:")
                print(input_sentence)
                print(f"Expected Output Sentence {print_count + 1}:")
                print(expected_sentence)
                print(f"Model Output Sentence {print_count + 1}:")
                print(output_sentence)
                print("=" * 50)
                print_count += 1

    avg_loss = total_loss / len(test_loader)
    print(f"Average Loss on Test Set: {avg_loss}")


# check opt output
def check_opt_hidden_state(prompt, opt_model, opt_tokenizer, opt_layer):
    # Tokenize the input prompt
    opt_inputs = opt_tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=max_length)
    prompt_length = len(opt_inputs['input_ids'][0])

    # Extract the attention mask
    attention_mask = opt_inputs['attention_mask']

    # Generate hidden states and continuation tokens for the given prompt
    with torch.no_grad():
        opt_outputs = opt_model(**opt_inputs, output_hidden_states=True)
    hidden_states = opt_outputs.hidden_states  # List of tensors for each layer
    opt_hidden_state = hidden_states[opt_layer]  # Extract the specified layer's hidden states

    # Get the length of the input sentence tokens
    non_padding_length = opt_inputs.input_ids.size(1)

    # Extract hidden states up to the length of the input sentence tokens
    opt_hidden_state = opt_hidden_state[:, :non_padding_length, :]

    # Pad the hidden states manually to ensure the size is 5
    padding_length = max_length - non_padding_length
    if padding_length > 0:
        padding_tensor = torch.zeros((opt_hidden_state.size(0), padding_length, opt_hidden_state.size(2)))
        opt_hidden_state = torch.cat([opt_hidden_state, padding_tensor], dim=1)

    # Generate continuation tokens

    generated_ids = opt_model.generate(opt_inputs.input_ids, max_new_tokens=prompt_length)
    generated_text = opt_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Tokenize the generated text
    generated_tokens = opt_tokenizer.tokenize(generated_text)

    print("Input Prompt: ", prompt)
    print("Generated Text: ", generated_text)
    print("All Tokens: ", generated_tokens)
    # print("Attention Mask: ", attention_mask)
    # print(f"Hidden States at Layer {opt_layer}: ", opt_hidden_state)

    return generated_text, attention_mask, opt_hidden_state


def prepare_data(df):
    data = []
    for i, row in df.iterrows():
        prompt = row['English sentence']
        print(f"Processing sentence {i + 1}/{len(df)}")

        if i > 3000:
            break

        # OPT last layer
        opt_inputs = opt_tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=max_length)

        # Generate hidden states and continuation tokens for the given prompt
        with torch.no_grad():
            opt_outputs = opt_model(**opt_inputs, output_hidden_states=True)
        hidden_states = opt_outputs.hidden_states  # List of tensors for each layer
        opt_hidden_state = hidden_states[opt_layer]  # Extract the specified layer's hidden states

        # Convert input_ids to tokens for filtering
        tokens = opt_tokenizer.convert_ids_to_tokens(opt_inputs.input_ids[0].tolist())
        print(f"OPT Input Tokens: {tokens}")

        # Filter out `</s>` token and its corresponding hidden state
        filtered_indices = [i for i, token in enumerate(tokens) if token != '</s>']
        # print(f"Filtered OPT Input Tokens: {filtered_indices}")
        opt_hidden_state = opt_hidden_state[:, filtered_indices, :]

        # Pad the hidden states manually to ensure the size is max_length
        non_padding_length = opt_hidden_state.size(1)
        padding_length = max_length - non_padding_length
        if padding_length > 0:
            padding_tensor = torch.zeros((opt_hidden_state.size(0), padding_length, opt_hidden_state.size(2)))
            opt_hidden_state = torch.cat([opt_hidden_state, padding_tensor], dim=1)

        # Translator first layer
        translator_inputs = translator_tokenizer(prompt, return_tensors="pt", padding=False, truncation=True,
                                                 max_length=max_length).to(device)

        decoder_start_token_id = translator_tokenizer.pad_token_id
        decoder_input_ids = torch.full((translator_inputs.input_ids.size(0), 1), decoder_start_token_id,
                                       dtype=torch.long).to(device)

        with torch.no_grad():
            # Generate the outputs from the model
            translator_outputs = translator_model(input_ids=translator_inputs.input_ids,
                                                  attention_mask=translator_inputs.attention_mask,
                                                  decoder_input_ids=decoder_input_ids,
                                                  output_hidden_states=True)

        hidden_states = translator_outputs.encoder_hidden_states
        translator_hidden_state = hidden_states[translator_layer]

        # Convert input_ids to tokens for filtering
        input_tokens = translator_tokenizer.convert_ids_to_tokens(translator_inputs.input_ids[0].tolist())
        print(f"Translator Input Tokens: {input_tokens}")

        # Filter out `</s>` token and its corresponding hidden state
        filtered_indices = [i for i, token in enumerate(input_tokens) if token != '</s>']
        # print(f"Filtered Translator Input Tokens: {filtered_indices}")
        translator_hidden_state = translator_hidden_state[:, filtered_indices, :]

        # Pad the hidden states manually to ensure the size is max_length
        non_padding_length = translator_hidden_state.size(1)
        padding_length = max_length - non_padding_length
        if padding_length > 0:
            padding_tensor = torch.zeros(
                (translator_hidden_state.size(0), padding_length, translator_hidden_state.size(2))).to(device)
            translator_hidden_state = torch.cat([translator_hidden_state, padding_tensor], dim=1)

        # Add the padded and truncated data
        data.append((opt_hidden_state, translator_hidden_state))

        # Explicitly clear variables to free memory
        del opt_inputs, opt_outputs, translator_inputs, translator_outputs
        torch.cuda.empty_cache()

    # Split data into train, validation, and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # Save data
    torch.save((train_data, val_data, test_data), 'data2.pt')


def find_divisors(n):
    divisors = []
    for i in range(1, n + 1):
        if n % i == 0:
            divisors.append(i)
    return divisors


# Objective function for Optuna
def objective(trial):
    num_layers = trial.suggest_int('num_layers', 1, 12)
    dim_feedforward = trial.suggest_int('dim_feedforward', 128, 512)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3)

    # Ensure num_heads is a divisor of input_dim
    divisors = find_divisors(input_dim)
    if not divisors:
        raise ValueError(f"No valid divisors found for input_dim={input_dim}. Please check the input dimension.")

    num_heads = trial.suggest_categorical('num_heads', divisors)

    print(f"Trial {trial.number}: num_layers={num_layers}, dim_feedforward={dim_feedforward}, "
          f"dropout={dropout}, learning_rate={learning_rate}, num_heads={num_heads}")

    model, val_loss = train_transformer(
        train_loader,
        val_loader,
        input_dim,
        output_dim,
        num_layers,
        num_heads,
        dim_feedforward,
        dropout,
        epochs=10,
        learning_rate=learning_rate
    )

    return val_loss


def optuna_fun(train_loader, val_loader, input_dim, output_dim):
    # Ensure train_loader and val_loader are defined before running the Optuna study
    if 'train_loader' in globals() and 'val_loader' in globals():
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)

        # Train and evaluate the final model with the best hyperparameters
        best_params = study.best_params
        print(f"Best Hyperparameters: {best_params}")

        # Train the final model with the best hyperparameters
        final_model, final_val_loss = train_transformer(
            train_loader,
            val_loader,
            input_dim,
            output_dim,
            best_params['num_layers'],
            best_params['num_heads'],
            best_params['dim_feedforward'],
            best_params['dropout'],
            epochs=10,
            learning_rate=best_params['learning_rate'])

        torch.save(final_model, 'best_model.pth')
    else:
        print("train_loader or val_loader not defined. Please ensure data loaders are correctly initialized.")


if __name__ == '__main__':
    # Load data
    data_path = 'C:\\Users\\talia\\PycharmProjects\\HebrewLLM\\translated_wikipedia_data.csv'
    df = pd.read_csv(data_path)
    df['English sentence'] = df['English sentence'].astype(str)

    # Prepare data
    # prepare_data(df)

    # Load data
    train_data, val_data, test_data = torch.load('data2.pt')

    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True,
                              collate_fn=custom_collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=batch_size, shuffle=False,
                             collate_fn=custom_collate_fn, num_workers=4, pin_memory=True)
    print("Data Loaded!\n")

    # Train the model
    input_dim = 512  # opt_hidden_state.shape[-1]
    output_dim = 512  # translator_hidden_state.shape[-1]
    hidden_dim = 128

    # fine tuning
    # optuna_fun(train_loader, val_loader, input_dim, output_dim)

    model = torch.load('best_model.pth')
    print("Model Loaded!\n")

    # criterion = nn.MSELoss()
    # evaluate_model(model, test_loader, criterion, num_prints=5)

    # Check prompt and result
    print("--------- CHECK ----------")
    prompt = "Good luck my "

    # take first 5 tokens of OPT hidden states
    generated_text, attention_mask, hidden_states = check_opt_hidden_state(prompt, opt_model, opt_tokenizer, opt_layer)

    hidden_states = model(hidden_states)

    # print("Converted hidden_states size: ", hidden_states.shape)

    sliced_hidden_states = hidden_states[:, :max_length, :]

    # print("sliced hidden_states size: ", sliced_hidden_states.shape)
    # print("Converted hidden_states: ", sliced_hidden_states)

    translator_inputs = translator_tokenizer(prompt, return_tensors="pt", padding=False, truncation=True,
                                             max_length=max_length).to(device)

    decoder_start_token_id = translator_tokenizer.pad_token_id
    decoder_input_ids = torch.full((translator_inputs.input_ids.size(0), 1), decoder_start_token_id,
                                   dtype=torch.long).to(device)

    with torch.no_grad():
        # Generate the outputs from the model
        translator_outputs = translator_model(input_ids=translator_inputs.input_ids,
                                              attention_mask=translator_inputs.attention_mask,
                                              decoder_input_ids=decoder_input_ids,
                                              output_hidden_states=True)

    hidden_states = translator_outputs.encoder_hidden_states
    translator_hidden_state = hidden_states[translator_layer]

    # Get the length of the input sentence tokens
    non_padding_length = translator_inputs.input_ids.size(1)

    # Extract hidden states up to the length of the input sentence tokens
    translator_hidden_state = translator_hidden_state[:, :non_padding_length, :]

    # Pad the hidden states manually to ensure the size is max_length
    padding_length = max_length - non_padding_length
    if padding_length > 0:
        padding_tensor = torch.zeros(
            (translator_hidden_state.size(0), padding_length, translator_hidden_state.size(2))).to(device)
        translator_hidden_state = torch.cat([translator_hidden_state, padding_tensor], dim=1)

    # print("Translator hidden_state: ", translator_hidden_state)

    # Call the function with hidden_states and attention_mask
    layer = 1
    outputs, res_generated_text = translator_activation_different_layer(hidden_states=sliced_hidden_states,
                                                                        attention_mask=attention_mask,
                                                                        nlayer=layer,
                                                                        max_length=max_length)

    print("Word: " + prompt)
    print("Result: " + res_generated_text)
