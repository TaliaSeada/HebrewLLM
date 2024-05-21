import torch
import torch.nn as nn
from transformers import AutoTokenizer, OPTForCausalLM, AutoModelForSeq2SeqLM
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from embeddingsToTranslator import translator_activation_different_layer
import torch.nn.functional as F
import optuna
import math

# opt
model_to_use = "350m"
opt_model = OPTForCausalLM.from_pretrained("facebook/opt-" + model_to_use)
opt_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-" + model_to_use)
opt_layer = -1
device = "cuda" if torch.cuda.is_available() else "cpu"
# translator
model_name = "Helsinki-NLP/opus-mt-en-he"
translator_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
translator_tokenizer = AutoTokenizer.from_pretrained(model_name)
translator_layer = 1


# transformer
# class HiddenStateTransformer(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=10, dropout=0.3):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.num_layers = num_layers
#
#         layers = []
#         for _ in range(num_layers):
#             layers.append(nn.Linear(input_dim, hidden_dim))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(dropout))
#             input_dim = hidden_dim
#         layers.append(nn.Linear(hidden_dim, output_dim))
#         self.network = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.network(x)
class HiddenStateTransformer(nn.Module):
    def __init__(self, input_size, output_size, num_layers, num_heads, dim_feedforward, dropout=0.1):
        super(HiddenStateTransformer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Transformer Encoder Layer
        encoder_layers = TransformerEncoderLayer(d_model=input_size, nhead=num_heads,
                                                 dim_feedforward=dim_feedforward, dropout=dropout, activation=F.relu)
        encoder_layers.self_attn.batch_first = True
        # encoder_layers.activation_relu_or_gelu=True
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers,
                                                      enable_nested_tensor=1 - (num_heads % 2))

        # Linear layer to map to the target hidden size
        self.fc = nn.Linear(input_size, 512)

    def forward(self, src):
        # src shape: (seq_length, batch_size, input_size)
        encoded = self.transformer_encoder(src)
        # encoded shape: (seq_length, batch_size, input_size)
        # Apply ReLU activation function
        # activated = F.relu(encoded)
        output = self.fc(encoded)
        # output shape: (seq_length, batch_size, output_size)
        return output


# class HiddenStateTransformer(nn.Module):
#     def __init__(self, input_dim, output_dim, num_layers=1, hidden_dim=128, num_heads=8, dropout=0.1):
#         super(HiddenStateTransformer, self).__init__()
#
#         self.embedding = nn.Linear(input_dim, hidden_dim)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x):
#         x = self.embedding(x)
#         x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, features)
#         x = self.transformer_encoder(x)
#         x = x.permute(1, 0, 2)  # Back to (batch, seq_len, features)
#         x = self.fc(x)  # Use only the last hidden state
#         return x


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

# Training function with gradient accumulation
# def train_transformer(train_loader, val_loader, input_size, output_size, epochs=10, learning_rate=0.01,
#                       accumulate_gradients_every=1):
#     # model = HiddenStateTransformer(input_dim=input_dim, output_dim=output_dim)
#     model = HiddenStateTransformer(input_size, output_size, num_layers, num_heads, dim_feedforward, dropout)
#     criterion = nn.MSELoss()
#     # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
#
#     accumulate_steps = 0
#     total_loss = 0
#     # model = torch.load('model_entire_layer.pth')
#     for epoch in range(epochs):
#         model.train(True)
#         for X_batch, Y_batch in train_loader:
#             optimizer.zero_grad()
#             outputs = model(X_batch)
#             loss = criterion(outputs, Y_batch)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#             accumulate_steps += 1
#
#             # Perform optimization step after accumulating gradients for specified number of steps
#             if accumulate_steps == accumulate_gradients_every:
#                 # optimizer.step()
#                 print(f"Epoch {epoch + 1}, Batch Loss: {total_loss / accumulate_steps}")
#                 accumulate_steps = 0
#                 total_loss = 0
#                 torch.save(model, 'model_entire_layer.pth')
#
#         # Validation phase
#         model.eval()
#         with torch.no_grad():
#             val_loss = 0
#             for X_batch, Y_batch in val_loader:
#                 outputs = model(X_batch)
#                 loss = criterion(outputs, Y_batch)
#                 val_loss += loss.item()
#
#             print(f"Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader)}")
#
#     # Save the entire model
#     # torch.save(model, 'model_entire_layer0.pth')
#     # torch.save(model, 'model_entire_layer1.pth')
#     return model


# Training function
# Modified training function with label smoothing
def train_transformer(train_loader, val_loader, input_size, output_size, num_layers, num_heads, dim_feedforward,
                      dropout, epochs=10, learning_rate=0.01, noise_factor=0.1, accumulate_gradients_every=1):
    model = HiddenStateTransformer(input_size, output_size, num_layers, num_heads, dim_feedforward, dropout)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

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

    return model, val_loss


def find_divisors(n):
    divisors = []
    for i in range(1, n + 1):
        if n % i == 0:
            divisors.append(i)
    return divisors


# Objective function for Optuna
def objective(trial):
    num_layers = trial.suggest_int('num_layers', 1, 4)
    # dim_model = trial.suggest_int('dim_model', 512, 1024)
    dim_feedforward = trial.suggest_int('dim_feedforward', 64, 4096)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)

    # Ensure num_heads is a divisor of input_dim
    divisors = find_divisors(input_dim)
    num_heads = trial.suggest_categorical('num_heads', divisors)

    model, val_loss = train_transformer(train_loader, val_loader, input_dim, output_dim, num_layers, num_heads,
                                        dim_feedforward, dropout, epochs=20, learning_rate=learning_rate)

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


if __name__ == '__main__':
    num_layers = 2  # Example, you can tune this
    num_heads = 1  # Example, you can tune this
    dim_feedforward = 128  # Example, you can tune this
    dropout = 0.1  # Example, you can tune this
    # Load data
    # data_path = 'C:\\Users\\talia\\PycharmProjects\\HebrewLLM\\resources\\dict.csv'
    # data_path = 'C:\\Users\\talia\\PycharmProjects\\HebrewLLM\\English_Hebrew_one_token.csv'
    # data_path = '/home/ubuntu/PycharmProjects/HebrewLLM/English_one_token.csv'
    data_path = 'C:\\Users\\talia\\PycharmProjects\\HebrewLLM\\English_one_token.csv'
    df = pd.read_csv(data_path)
    # df['translation'] = df['translation'].astype(str)
    df['English'] = df['English'].astype(str)

    # Prepare data
    # data = []
    # for i, row in df.iterrows():
    #     # if i > 500:
    #     #     break
    #     # prompt = row['translation']
    #     prompt = row['English']
    #
    #     # OPT last layer
    #     opt_inputs = opt_tokenizer(prompt, return_tensors="pt")
    #     opt_outputs = opt_model(**opt_inputs, output_hidden_states=True)
    #     opt_hidden_state = opt_outputs.hidden_states[opt_layer]
    #
    #     # Translator first layer
    #     translator_inputs = translator_tokenizer(prompt, return_tensors="pt")
    #     decoder_start_token_id = translator_tokenizer.pad_token_id
    #     decoder_input_ids = torch.full((translator_inputs.input_ids.size(0), 1), decoder_start_token_id,
    #                                    dtype=torch.long)
    #     translator_outputs = translator_model(input_ids=translator_inputs.input_ids,
    #                                           decoder_input_ids=decoder_input_ids, output_hidden_states=True)
    #     translator_hidden_state = translator_outputs.encoder_hidden_states[translator_layer]
    #
    #     # # Filter out long words
    #     # if opt_hidden_state.shape[1] != 2 or translator_hidden_state.shape[1] != 2:
    #     #     continue
    #     # a = [opt_hidden_state[0][1]]
    #     # data.append((a, translator_hidden_state))
    #     data.append((opt_hidden_state, translator_hidden_state))
    #
    # # Split data into train, validation, and test sets
    # train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    # train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    #
    # # Save data
    # torch.save((train_data, val_data, test_data), 'data1.pt')

    # Load data
    train_data, val_data, test_data = torch.load('data1.pt')

    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True,
                              collate_fn=custom_collate_fn)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=batch_size, shuffle=False,
                             collate_fn=custom_collate_fn)

    # Train the model
    input_dim = 512  #opt_hidden_state.shape[-1]
    output_dim = 512  #translator_hidden_state.shape[-1]
    hidden_dim = 128

    # # Train the model with gradient accumulation
    # model = train_transformer(train_loader, val_loader, input_dim, output_dim, epochs=15, learning_rate=0.00001,
    #                           accumulate_gradients_every=15)

    # Optimize hyperparameters with Optuna
    # study = optuna.create_study(direction='minimize')
    # study.optimize(objective, n_trials=50)
    #
    # print("Best hyperparameters:", study.best_params)
    #
    # # Train the model with the best hyperparameters
    # best_params = study.best_params
    # model, _ = train_transformer(train_loader, val_loader, input_dim, output_dim, best_params['num_layers'],
    #                              best_params['num_heads'], best_params['dim_feedforward'], best_params['dropout'],
    #                              epochs=20, learning_rate=best_params['learning_rate'])
    # torch.save(model, 'best_model.pth')

    # Evaluate the model
    model = torch.load('best_model.pth')
    model.eval()
    # criterion = nn.MSELoss()
    # evaluate_model(model, test_loader, criterion)

    print("--------- CHECK ----------")
    prompt = "can"
    opt_inputs = opt_tokenizer(prompt, return_tensors="pt")
    opt_outputs = opt_model(**opt_inputs, output_hidden_states=True)
    opt_hidden_state = opt_outputs.hidden_states[opt_layer]

    hidden_states = model(opt_hidden_state)

    layer = 1
    outputs, generated_text = translator_activation_different_layer(hidden_states, layer)
    print("Word: " + prompt)
    print("Result: " + generated_text)
