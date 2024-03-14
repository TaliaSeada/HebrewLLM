import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import AutoTokenizer, OPTForCausalLM, AutoModelForSeq2SeqLM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from embeddingsToTranslator import translator_activation_different_layer

data_path = 'C:\\Users\\talia\\PycharmProjects\\TranslatorGPT\\resources\\dict.csv'
df = pd.read_csv(data_path)
remove_period = True

# opt
model_to_use = "350m"
opt_model = OPTForCausalLM.from_pretrained("facebook/opt-" + model_to_use)
opt_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-" + model_to_use)
opt_layer = -1

# translator
model_name = "Helsinki-NLP/opus-mt-en-he"
translator_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
translator_tokenizer = AutoTokenizer.from_pretrained(model_name)
translator_layer = 1

X = []
y = []

for i, row in df.iterrows():
    prompt = row['translation']

    # OPT last layer
    opt_inputs = opt_tokenizer(prompt, return_tensors="pt")
    opt_outputs = opt_model(**opt_inputs, output_hidden_states=True)
    opt_hidden_state = opt_outputs.hidden_states[opt_layer]

    # translator first layer
    translator_inputs = translator_tokenizer(prompt, return_tensors="pt")
    decoder_start_token_id = translator_tokenizer.pad_token_id
    decoder_input_ids = torch.full((translator_inputs.input_ids.size(0), 1), decoder_start_token_id, dtype=torch.long).to(
        translator_inputs.input_ids.device)

    translator_outputs = translator_model(input_ids=translator_inputs.input_ids, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
    translator_hidden_state = translator_outputs.encoder_hidden_states[translator_layer]

    # we want for now only the short words
    if opt_hidden_state.shape[1] != 2 or translator_hidden_state.shape[1] != 2:
        continue

    X.append(opt_hidden_state)
    y.append(translator_hidden_state)

    # if len(X) > 4:
    #     break


# transformer
class HiddenStateTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[512, 512]):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_layers + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# training
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
# Custom Dataset to handle lists of tensors
class TensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Example of verifying and preparing tensors before padding
def custom_collate_fn(batch):
    X_batch, y_batch = zip(*batch)
    X_padded = pad_sequence([x.squeeze(0) for x in X_batch], batch_first=True, padding_value=0)
    y_padded = pad_sequence([y.squeeze(0) for y in y_batch], batch_first=True, padding_value=0)
    return X_padded, y_padded

torch.autograd.set_detect_anomaly(True)
# Revised training function
def train_transformer(X_train, X_val, Y_train, Y_val, input_dim, output_dim, epochs=10, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HiddenStateTransformer(input_dim=input_dim, output_dim=output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # try SGD
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Create DataLoader instances for training and validation sets
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()  # Ensure gradients are zeroed correctly here
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward(retain_graph=True)
            optimizer.step()

            total_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}, Training Loss: {total_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}")

    # Save the entire model
    torch.save(model, 'model_entire.pth')
    return model

def evaluate_model(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Average Loss on Test Set: {avg_loss}")



if __name__ == '__main__':
    # # Split data into training, validation, and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    #
    # # print("Training set size:", len(X_train))
    # # print("Validation set size:", len(X_val))
    # # print("Test set size:", len(X_test))
    #
    # print("--------- TRAINING ----------")
    # input_dim = 512
    # output_dim = 512
    # train_transformer(X_train, X_val, y_train, y_val, input_dim, output_dim, epochs=10)
    #
    # test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False,
    #                          collate_fn=custom_collate_fn)
    # criterion = nn.MSELoss()
    #
    # print("--------- TEST ----------")
    model = torch.load('model_entire.pth')
    model.eval()
    # evaluate_model(model, test_loader, criterion)

    print("--------- CHECK ----------")
    prompt = "Image"
    opt_inputs = opt_tokenizer(prompt, return_tensors="pt")
    opt_outputs = opt_model(**opt_inputs, output_hidden_states=True)
    opt_hidden_state = opt_outputs.hidden_states[opt_layer]

    hidden_states = model(opt_hidden_state)

    layer = 1
    outputs, generated_text = translator_activation_different_layer(hidden_states, layer)

    print(generated_text)

