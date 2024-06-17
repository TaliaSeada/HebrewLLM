import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModel, AutoTokenizer, OPTForCausalLM
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import embeddingsToOPT
from embeddingsToOPT import CustomLayerWrapper

# load data
# activateTrans1(0,0)


file = "C:\\Users\\relwe\\OneDrive\\Documents\\GitHub\\HebrewLLM\\resources\\He_En_oneTokenData.csv"
df = pd.read_csv(file)
df = df.dropna()
# build models
device = "cuda" if torch.cuda.is_available() else "cpu"
src_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
tgt_model_name = "facebook/opt-350m"
translator_He_En_model = AutoModelForSeq2SeqLM.from_pretrained(src_model_name).to(device)
# OPT_model = AutoModel.from_pretrained(tgt_model_name).to(device)
OPT_model = OPTForCausalLM.from_pretrained(tgt_model_name).to(device)

# Set input_size and output_size based on the model configurations
input_size = 3072 #translator_He_En_model.config.hidden_size
# print(input_size)
output_size = 1024 #OPT_model.config.hidden_size
# print(output_size)

# Tokenize sentences
translator_He_En_tokenizer = AutoTokenizer.from_pretrained(src_model_name)
OPT_tokenizer = AutoTokenizer.from_pretrained(tgt_model_name)

   
# Transformer
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
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, src):
        # src shape: (seq_length, batch_size, input_size)
        encoded = self.transformer_encoder(src)
        # encoded shape: (seq_length, batch_size, input_size)
        # Apply ReLU activation function
        # activated = F.relu(encoded)
        output = self.fc(encoded)
        # output shape: (seq_length, batch_size, output_size)
        return output


def train_model(train_loader, hidden_state_transformer, optimizer, criterion, num_epochs):
    total_loss = 0
    for X_batch, Y_batch in train_loader:
        mean_values = torch.mean(Y_batch, dim=0)
        # Expand the mean values back to the original shape
        avg_hs = mean_values.unsqueeze(0).expand_as(Y_batch)
        loss = criterion(avg_hs, Y_batch)
        total_loss += loss.item()
    total_loss /= len(train_loader)
    print("The loss on the Avg. value is: ", total_loss)
    for epoch in range(num_epochs):
        hidden_state_transformer.train()
        total_loss = 0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            predicted_value = hidden_state_transformer(X_batch)
            loss = criterion(predicted_value, Y_batch)
            total_loss += loss.item()
            # Backpropagation
            loss.backward()
            optimizer.step()

        total_loss /= len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}")


def infer_hidden_states(hidden_state_transformer, src_model, src_tokenizer, input_word, device):
    # src_input = src_inputs['input_ids'][2].unsqueeze(0).to(device)

    # # Forward pass through the source model to get hidden states
    # with torch.no_grad():
    #     src_output = src_model.generate(src_input, output_hidden_states=True,
    #                                 return_dict_in_generate=True, max_length=max_length)
    #     src_hidden_states = src_output.encoder_hidden_states[-1]  # Last layer's hidden states
    # Tokenize the input word
    tokenized_input = src_tokenizer.encode(input_word, return_tensors="pt", truncation=True, padding=True).to(device)

    # Forward pass through the source model to get hidden states
    with torch.no_grad():
        src_output = src_model.generate(tokenized_input, output_hidden_states=True, return_dict_in_generate=True,
                                        max_length=2)
        src_hidden_states = src_output.encoder_hidden_states[-1][0, :, :]  # Last layer's hidden states

        # print(src_hidden_states.shape)
        # print(src_hidden_states)

    # Forward pass through your transformer model
    with torch.no_grad():
        converted_hidden_states = hidden_state_transformer(src_hidden_states)

    return converted_hidden_states


# Function to load the model
def load_hidden_state_transformer(model_path, input_size, output_size, num_layers, num_heads, dim_feedforward, device):
    model = HiddenStateTransformer(input_size, output_size, num_heads, num_layers, dim_feedforward)
    
    model.load_state_dict(torch.load(model_path, map_location=device),strict=False)
    model.to(device)

    # Set the model to evaluation mode
    model.eval()
    return model

class TensorDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def custom_collate_fn(batch):
    X_batch, Y_batch = zip(*batch)
    X_padded = pad_sequence([x.squeeze(0) for x in X_batch], batch_first=True, padding_value=0)
    Y_padded = pad_sequence([y.squeeze(0) for y in Y_batch], batch_first=True, padding_value=0)
    return X_padded, Y_padded

def main():
    model_save_path = "hidden_state_transformer2500.pth"

    # Model parameters
    num_layers = 2  # Example, you can tune this
    num_heads = 1  # Example, you can tune this
    dim_feedforward = 128  # Example, you can tune this
    dropout = 0.1  # Example, you can tune this

    # Initialize the transformer model
    hidden_state_transformer = HiddenStateTransformer(input_size, output_size, num_layers, num_heads, dim_feedforward, dropout)
    data = torch.load('C:\\Users\\relwe\\OneDrive\\Documents\\GitHub\\HebrewLLM\\transformer_1\\hidden_state_transformer2500.pth')
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    # Create data loaders
    batch_size = 32  # Reduce batch size for memory optimization
    train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True,
                              collate_fn=custom_collate_fn)

    hidden_state_transformer.train()

    # Initialize the loss function and optimizer
    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    # criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(hidden_state_transformer.parameters(), lr=0.001)  # You can tune the learning rate
    # optimizer = torch.optim.SGD(hidden_state_transformer.parameters(), lr=0.1, momentum=0.9)
    num_epochs = 10  # Set the number of epochs for training

    # hidden_state_transformer = load_hidden_state_transformer(model_save_path, input_size, output_size, num_layers,
    #                                                          num_heads, dim_feedforward, dropout, device)
    # Call the training function
    train_model(train_loader, hidden_state_transformer, optimizer, criterion, num_epochs)
    # Saving the weights of the HiddenStateTransformer
    
    torch.save(hidden_state_transformer.state_dict(), model_save_path)
    # torch.save(hidden_state_transformer, model_save_path)

    # Loading the model
    hidden_state_transformer = load_hidden_state_transformer(model_save_path, input_size, output_size, num_layers,
                                                             num_heads, dim_feedforward, device)
    
    text = "הולך"
    translator_inputs = translator_He_En_tokenizer.encode_plus(text, return_tensors='pt', truncation=True)
    translator_outputs = translator_He_En_model.generate(input_ids=translator_inputs.input_ids,
                                                              attention_mask=translator_inputs.attention_mask,
                                                              # decoder_input_ids=decoder_input_ids,
                                                              # max_length=16,  # adjust max_length as needed
                                                              num_beams=5,  # adjust num_beams as needed
                                                              early_stopping=True,
                                                              output_hidden_states=True)
    translator_outputs1 = translator_He_En_model(input_ids=translator_inputs.input_ids,
                                              decoder_input_ids=translator_outputs, output_hidden_states=True)
    translator_hidden_state = translator_outputs1.decoder_hidden_states[-1]
    # OPT_hidden_state = torch.empty(1,2,1024)
    OPT_hidden_state = hidden_state_transformer(translator_hidden_state)
    outputs, generated_text = embeddingsToOPT.OPT_activation_different_layer(OPT_hidden_state, 0)
    print(generated_text)
    
if __name__ == '__main__':
    main()