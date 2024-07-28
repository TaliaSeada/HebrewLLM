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

# opt
model_to_use = "350m"
opt_model = OPTForCausalLM.from_pretrained("facebook/opt-" + model_to_use)
opt_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-" + model_to_use, padding_side='left')
opt_layer = -1
device = "cuda" if torch.cuda.is_available() else "cpu"
opt_model.to(device)

# translator
model_name = "Helsinki-NLP/opus-mt-en-he"
translator_tokenizer = MarianTokenizer.from_pretrained(model_name)
translator_model = MarianMTModel.from_pretrained(model_name)
translator_layer = 1
translator_model.to(device)


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

        # Linear layer to map to the target hidden size
        self.fc = nn.Linear(input_size, 512)

    def forward(self, src):
        # src shape: (seq_length, batch_size, input_size)
        encoded = self.transformer_encoder(src)
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
    return targets + noise


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


def find_divisors(n):
    divisors = []
    for i in range(1, n + 1):
        if n % i == 0:
            divisors.append(i)
    return divisors


# Objective function for Optuna
def objective(trial):
    num_layers = trial.suggest_int('num_layers', 1, 8)
    dim_feedforward = trial.suggest_int('dim_feedforward', 64, 4096)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3)

    # Ensure num_heads is a divisor of input_dim
    divisors = find_divisors(input_dim)
    num_heads = trial.suggest_categorical('num_heads', divisors)

    model, val_loss = train_transformer(
        train_loader,
        val_loader,
        input_dim,
        output_dim,
        num_layers,
        num_heads,
        dim_feedforward,
        dropout,
        epochs=20,
        learning_rate=learning_rate)

    return val_loss


# Evaluation function
def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Average Loss on Test Set: {avg_loss}")


# check opt output
def check_opt_hidden_state(prompt, opt_model, opt_tokenizer, opt_layer):
    # Tokenize the input prompt
    opt_inputs = opt_tokenizer(prompt, return_tensors="pt", padding='max_length', truncation=True, max_length=15)

    # Extract the attention mask
    attention_mask = opt_inputs['attention_mask']

    # Generate hidden states for the given prompt
    opt_outputs = opt_model(**opt_inputs, output_hidden_states=True)
    hidden_states = opt_outputs.hidden_states  # List of tensors for each layer
    opt_hidden_state = hidden_states[opt_layer]  # Extract the specified layer's hidden states

    # Generate continuation tokens
    generated_ids = opt_model.generate(opt_inputs.input_ids, max_new_tokens=15)
    generated_text = opt_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Tokenize the generated text
    generated_tokens = opt_tokenizer.tokenize(generated_text)

    # Convert tokens to a single string
    all_tokens_text = opt_tokenizer.convert_tokens_to_string(generated_tokens)

    print("Input Prompt: ", prompt)
    print("Generated Text: ", generated_text)
    print("All Tokens: ", generated_tokens)
    print("Tokens as Text: ", all_tokens_text)
    print("Attention Mask: ", attention_mask)
    print(f"Hidden States at Layer {opt_layer}: ", opt_hidden_state)

    return generated_text, attention_mask, opt_hidden_state


if __name__ == '__main__':
    # num_layers = 2
    # num_heads = 1
    # dim_feedforward = 128
    # dropout = 0.1

    # Load data
    data_path = 'C:\\Users\\talia\\PycharmProjects\\HebrewLLM\\translated_wikipedia_data.csv'
    df = pd.read_csv(data_path)
    df['English sentence'] = df['English sentence'].astype(str)

    # Prepare data
    # max_length = 15
    # data = []
    # for i, row in df.iterrows():
    #     prompt = row['English sentence']
    #     print(i)
    #     if i > 20000:
    #         break
    #
    #     # OPT last layer
    #     opt_inputs = opt_tokenizer(prompt, return_tensors="pt", padding='max_length', truncation=True,
    #                                max_length=max_length).to(device)
    #     opt_outputs = opt_model(**opt_inputs, output_hidden_states=True)
    #     opt_hidden_state = opt_outputs.hidden_states[opt_layer].detach().cpu()
    #
    #     # Translator first layer
    #     translator_inputs = translator_tokenizer(prompt, return_tensors="pt", padding='max_length', truncation=True,
    #                                              max_length=max_length).to(device)
    #     decoder_start_token_id = translator_tokenizer.pad_token_id
    #     decoder_input_ids = torch.full((translator_inputs.input_ids.size(0), 1), decoder_start_token_id,
    #                                    dtype=torch.long).to(device)
    #     translator_outputs = translator_model(input_ids=translator_inputs.input_ids,
    #                                           attention_mask=translator_inputs.attention_mask,
    #                                           decoder_input_ids=decoder_input_ids, output_hidden_states=True)
    #     translator_hidden_state = translator_outputs.encoder_hidden_states[translator_layer].detach().cpu()
    #
    #     # Add the padded and truncated data
    #     data.append((opt_hidden_state, translator_hidden_state))
    #
    #     # Explicitly clear variables to free memory
    #     del opt_inputs, opt_outputs, translator_inputs, translator_outputs
    #     torch.cuda.empty_cache()
    #
    # # Split data into train, validation, and test sets
    # train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    # train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    #
    # # Save data
    # torch.save((train_data, val_data, test_data), 'data2.pt')

    # Load data
    train_data, val_data, test_data = torch.load('data2.pt')

    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True,
                              collate_fn=custom_collate_fn)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=batch_size, shuffle=False,
                             collate_fn=custom_collate_fn)

    # Train the model
    input_dim = 512  # opt_hidden_state.shape[-1]
    output_dim = 512  # translator_hidden_state.shape[-1]
    hidden_dim = 128

    # # Optimize hyperparameters with Optuna
    # study = optuna.create_study(direction='minimize')
    # study.optimize(objective, n_trials=10)
    #
    # print(f"Best trial: {study.best_trial.params}")
    # print("Best hyperparameters:", study.best_params)
    #
    # # Train the model with the best hyperparameters
    # best_trial = study.best_trial
    # best_model, _ = train_transformer(train_loader, val_loader, input_dim, output_dim,
    #                                   best_trial.params['num_layers'], best_trial.params['num_heads'],
    #                                   best_trial.params['dim_feedforward'], best_trial.params['dropout'],
    #                                   epochs=20, learning_rate=best_trial.params['learning_rate'])
    # torch.save(best_model, 'best_model.pth')

    model = torch.load('best_model.pth')
    #
    # criterion = nn.MSELoss()
    # evaluate_model(model, test_loader, criterion)

    # Check prompt and result
    print("--------- CHECK ----------")
    max_length = 15
    prompt = "Hello, my name is Talia and I love to"
    # opt_inputs = opt_tokenizer(prompt, return_tensors="pt", padding='max_length', truncation=True,
    #                            max_length=max_length)
    # opt_outputs = opt_model(**opt_inputs, output_hidden_states=True)
    # opt_hidden_state = opt_outputs.hidden_states[opt_layer]
    # attention_mask = opt_inputs['attention_mask']

    # take first 15 tokens of OPT hidden states
    num_of_tokens = 15
    generated_text, attention_mask, hidden_states = check_opt_hidden_state(prompt, opt_model, opt_tokenizer, opt_layer)

    hidden_states = model(hidden_states)

    print("Converted hidden_states size: ", hidden_states.shape)

    sliced_hidden_states = hidden_states[:, :num_of_tokens, :]

    print("sliced hidden_states size: ", hidden_states.shape)

    # Call the function with hidden_states and attention_mask
    layer = 1
    outputs, res_generated_text = translator_activation_different_layer(hidden_states=sliced_hidden_states,
                                                                        attention_mask=attention_mask,
                                                                        nlayer=layer,
                                                                        max_length=15)

    print("Word: " + prompt)
    print("Result: " + res_generated_text)
