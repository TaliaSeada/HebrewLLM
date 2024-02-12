import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import AutoTokenizer, OPTForCausalLM, AutoModelForSeq2SeqLM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
translator_layer = 0

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
    if opt_hidden_state.shape[1] != 2 and translator_hidden_state.shape[1] != 2:
        continue

    X.append(opt_hidden_state)
    y.append(translator_hidden_state)

    if len(X) > 4:
        break


# transformer
class HiddenStateTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[512, 512]):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_layers + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation function on the last layer
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# training
def train_transformer(X_train, X_val, Y_train, Y_val, input_dim, output_dim, epochs=10, batch_size=32, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HiddenStateTransformer(input_dim=input_dim, output_dim=output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Convert datasets to tensor and move to the correct device
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train_tensor[i:i+batch_size]
            Y_batch = Y_train_tensor[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i in range(0, len(X_val), batch_size):
                X_batch = X_val_tensor[i:i+batch_size]
                Y_batch = Y_val_tensor[i:i+batch_size]
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}, Training Loss: {total_loss / (len(X_train) / batch_size)}, Validation Loss: {val_loss / (len(X_val) / batch_size)}")

    return model


if __name__ == '__main__':
    # split
    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2)
    X
    input_dim = 512
    output_dim = 512
    train_transformer(X_train, X_val, Y_train, Y_val, input_dim, output_dim)



"""
we have:
 opt_hidden_state and translator_hidden_state
 * take only the "good" ones
 make test and train sets
 
we need:
 build a transformer
 build a training function
"""