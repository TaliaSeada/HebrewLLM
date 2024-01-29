import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModel, AutoTokenizer
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# load data
import embeddingsToOPT
# TODO train on the big data
df = pd.read_csv("C:\\Users\\talia\\PycharmProjects\\TranslatorGPT\\resources\\dict.csv")

# build models
device = "cuda" if torch.cuda.is_available() else "cpu"
src_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
tgt_model_name = "facebook/opt-350m"
src_model = AutoModelForSeq2SeqLM.from_pretrained(src_model_name).to(device)
tgt_model = AutoModel.from_pretrained(tgt_model_name).to(device)

# Set input_size and output_size based on the model configurations
input_size = src_model.config.hidden_size
output_size = tgt_model.config.hidden_size

# Tokenize sentences
src_tokenizer = AutoTokenizer.from_pretrained(src_model_name)
tgt_tokenizer = AutoTokenizer.from_pretrained(tgt_model_name)
max_length = 512  # Example length, adjust as needed
src_inputs = src_tokenizer.batch_encode_plus(df['statement'].tolist(), padding='max_length', max_length=max_length, return_tensors='pt', truncation=True).to(device)
tgt_inputs = tgt_tokenizer.batch_encode_plus(df['translation'].tolist(), padding='max_length', max_length=max_length, return_tensors='pt', truncation=True).to(device)


# Transformer
class HiddenStateTransformer(nn.Module):
    def __init__(self, input_size, output_size, num_layers, num_heads, dim_feedforward, dropout=0.1):
        super(HiddenStateTransformer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Transformer Encoder Layer
        encoder_layers = TransformerEncoderLayer(d_model=input_size, nhead=num_heads,
                                                 dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # Linear layer to map to the target hidden size
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, src):
        # src shape: (seq_length, batch_size, input_size)
        encoded = self.transformer_encoder(src)
        # encoded shape: (seq_length, batch_size, input_size)
        output = self.fc(encoded)
        # output shape: (seq_length, batch_size, output_size)
        return output


def train_model(src_model, tgt_model, hidden_state_transformer, src_inputs, tgt_inputs, optimizer, criterion, num_epochs, device):
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(src_inputs['input_ids'].shape[0]):
            # Prepare the inputs
            src_input = src_inputs['input_ids'][i].unsqueeze(0).to(device)
            tgt_input = tgt_inputs['input_ids'][i].unsqueeze(0).to(device)

            # Forward pass through the source model to get hidden states
            with torch.no_grad():
                src_output = src_model.generate(src_input, output_hidden_states=True,
                                                 return_dict_in_generate=True, max_length=512)
                # src_output = src_model(input_ids=src_input, output_hidden_states=True)
                src_hidden_states = src_output.encoder_hidden_states[-1]  # Last layer's hidden states

            # Forward pass through the target model to get target hidden states
            with torch.no_grad():
                tgt_output = tgt_model(input_ids=tgt_input, output_hidden_states=True)
                tgt_hidden_states = tgt_output.hidden_states[0]  # First layer's hidden states

            # Forward pass through your transformer model
            predicted_tgt_hidden_states = hidden_state_transformer(src_hidden_states)

            # Calculate loss
            # print("Predicted shape:", predicted_tgt_hidden_states.shape)
            # print("Target shape:", tgt_hidden_states.shape)

            loss = criterion(predicted_tgt_hidden_states, tgt_hidden_states)
            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(src_inputs['input_ids'])}")


def infer_hidden_states(hidden_state_transformer, src_model, src_tokenizer, input_word, device):
    # Tokenize the input word
    tokenized_input = src_tokenizer.encode(input_word, return_tensors="pt", truncation=True, padding=True).to(device)

    # Forward pass through the source model to get hidden states
    with torch.no_grad():
        src_output = src_model.generate(tokenized_input, output_hidden_states=True, return_dict_in_generate=True, max_length=512)
        src_hidden_states = src_output.encoder_hidden_states[-1]  # Last layer's hidden states

    # Forward pass through your transformer model
    with torch.no_grad():
        converted_hidden_states = hidden_state_transformer(src_hidden_states)

    return converted_hidden_states


# Function to load the model
def load_hidden_state_transformer(model_path, input_size, output_size, num_layers, num_heads, dim_feedforward, dropout, device):
    model = HiddenStateTransformer(input_size, output_size, num_layers, num_heads, dim_feedforward, dropout)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


def main():
    model_save_path = "hidden_state_transformer.pth"

    # Model parameters
    num_layers = 4  # Example, you can tune this
    num_heads = 8  # Example, you can tune this
    dim_feedforward = 2048  # Example, you can tune this
    dropout = 0.1  # Example, you can tune this

    # Initialize the transformer model
    hidden_state_transformer = HiddenStateTransformer(input_size, output_size, num_layers, num_heads, dim_feedforward, dropout).to(device)

    # Initialize the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(hidden_state_transformer.parameters(), lr=0.001)  # You can tune the learning rate
    num_epochs = 10  # Set the number of epochs for training
    # Call the training function
    train_model(src_model, tgt_model, hidden_state_transformer, src_inputs, tgt_inputs, optimizer, criterion, num_epochs, device)

    # Saving the weights of the HiddenStateTransformer
    torch.save(hidden_state_transformer.state_dict(), model_save_path)

    # Loading the model
    hidden_state_transformer = load_hidden_state_transformer(model_save_path, input_size, output_size, num_layers,
                                                             num_heads, dim_feedforward, dropout, device)
    # Example Hebrew word
    input_word = "שלום"  # Replace with your Hebrew word

    # Using the function
    converted_hidden_states = infer_hidden_states(hidden_state_transformer, src_model, src_tokenizer, input_word,
                                                  device)
    print(converted_hidden_states)

    # Continue with your OPT model or any other post-processing steps
    layer = 0
    outputs, generated_text = embeddingsToOPT.OPT_activation_different_layer(converted_hidden_states, layer)
    print("Generated Text: ", generated_text)

if __name__ == '__main__':
    main()




# import embeddingsToOPT
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
# from sklearn.model_selection import train_test_split
# from torch.utils import data
#
#
# # Load data
# df = pd.read_csv("C:\\Users\\talia\\PycharmProjects\\TranslatorGPT\\output_trans\\translate_short_words_1.csv")
#
# # Split data into training and validation sets
# train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# src_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
# tgt_model_name = "facebook/opt-350m"
# src_model = AutoModelForSeq2SeqLM.from_pretrained(src_model_name).to(device)
# tgt_model = AutoModel.from_pretrained(tgt_model_name).to(device)
#
# # Set input_size and output_size based on the model configurations
# input_size = src_model.config.hidden_size  # Adjust if necessary
# output_size = tgt_model.config.hidden_size  # Adjust if necessary
#
# # Tokenize sentences
# src_tokenizer = AutoTokenizer.from_pretrained(src_model_name)
# tgt_tokenizer = AutoTokenizer.from_pretrained(tgt_model_name)
#
# # Tokenize and pad sentences
# src_inputs = src_tokenizer(train_df['statement'].tolist(), padding=True, return_tensors='pt', truncation=True)
# tgt_inputs = tgt_tokenizer(train_df['translation'].tolist(), padding=True, return_tensors='pt', truncation=True)
#
# # Prepare data loaders
# train_dataset = data.TensorDataset(src_inputs['input_ids'], tgt_inputs['input_ids'])
# train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
#
#
# # Define the transformer model
# class WordTranslationTransformer(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(WordTranslationTransformer, self).__init__()
#         self.embedding = nn.Embedding(input_size, input_size)
#         self.transformer_encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=input_size, nhead=1),
#             num_layers=3
#         )
#         self.fc = nn.Linear(input_size, output_size)
#
#     def forward(self, x):
#         x = x.unsqueeze(0)  # Add a batch dimension
#         x = self.embedding(x)
#         x = x.permute(1, 0, 2)  # Change the order of dimensions
#         x = self.transformer_encoder(x)
#         x = x.permute(1, 0, 2)  # Change the order of dimensions back
#         x = self.fc(x)
#         return x.squeeze(0)  # Remove the added batch dimension
#
#
# def train_model():
#     # Initialize model
#     model = WordTranslationTransformer(input_size=input_size, output_size=output_size).to(device)
#
#     # Define loss and optimizer
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#     # Training loop
#     epochs = 10
#
#     for epoch in range(epochs):
#         total_loss = 0
#
#         for batch_src, batch_tgt in train_loader:
#             batch_src, batch_tgt = batch_src.to(device), batch_tgt.to(device)
#
#             # Forward pass
#             optimizer.zero_grad()
#             output_states = model(batch_src)
#
#             # Compute loss
#             loss = criterion(output_states, batch_tgt)
#             total_loss += loss.item()
#
#             # Backward pass and optimization
#             loss.backward()
#             optimizer.step()
#
#         # Print average loss for the epoch
#         avg_loss = total_loss / len(train_loader)
#         print(f"Epoch {epoch}, Average Loss: {avg_loss}")
#
#     # Save the trained model
#     torch.save(model.state_dict(), 'word_translation_transformer.pth')
#
# if __name__ == '__main__':
#     train_model()

    # # Initialize the WordTranslationTransformer model
    # model = WordTranslationTransformer(input_size=input_size, output_size=output_size).to(device)
    # # Load the saved model state dictionary
    # model.load_state_dict(torch.load('word_translation_transformer.pth'))
    # # Set the model to evaluation mode
    # model.eval()

    # new_hebrew_sentence = "חתול"
    # src_input = src_tokenizer.encode(new_hebrew_sentence, return_tensors='pt').squeeze(0).to(device)
    #
    # # Get the hidden states from the WordSpecificTransformer
    # transformed_states = model(src_input)
    #
    # # Continue with your OPT model or any other post-processing steps
    # layer = 1
    # outputs, generated_text = embeddingsToOPT.OPT_activation_different_layer(transformed_states, layer)
    # try using the result as input to the OPT model

    # outputs, generated_text = embeddingsToOPT.OPT_activation_different_layer(generate("חתול"), layer)
    # print("Generated Text: ", generated_text)


# import torch
# from torch import nn
# from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
#
# # Load the tokenizers and models
# import embeddingsToOPT
#
# device = "cpu"
# src_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
# tgt_model_name = "facebook/opt-350m"
# src_tokenizer = AutoTokenizer.from_pretrained(src_model_name)
# tgt_tokenizer = AutoTokenizer.from_pretrained(tgt_model_name)
# src_model = AutoModelForSeq2SeqLM.from_pretrained(src_model_name).to(device)
# tgt_model = AutoModel.from_pretrained(tgt_model_name)
#
# # Example input sentences
# tgt_input = "Hello"
# src_input = "שלום"
#
# # Tokenize and get hidden states from the last layer of the source model
# src_inputs = src_tokenizer.encode(src_input, return_tensors='pt').to(device)
# src_outputs = src_model.generate(src_inputs, output_hidden_states=True,
#                                  return_dict_in_generate=True, max_length=512)
# src_layer = -1
# src_hidden_states = src_outputs.encoder_hidden_states[src_layer].squeeze(0).to(device)
#
# # Tokenize and get hidden states from the first layer of the target model
# tgt_inputs = tgt_tokenizer(tgt_input, return_tensors="pt")
# tgt_outputs = tgt_model(**tgt_inputs, output_hidden_states=True)
# tgt_layer = 1
# tgt_hidden_states = tgt_outputs.hidden_states[tgt_layer].squeeze(0)
#
# # Ensure input_size matches the feature dimension of hidden states
# input_size = src_hidden_states.size(1)
# output_size = tgt_hidden_states.size(1)
#
#
# # Define the transformer model
# class TransformerModel(nn.Module):
#     def __init__(self, input_size, output_size):
#         super().__init__()
#         self.transformer = nn.Transformer(d_model=input_size, nhead=1)
#         self.fc = nn.Linear(input_size, output_size)
#
#     def forward(self, x):
#         x = self.transformer(x, x)
#         x = self.fc(x)
#         return x
#
#
# # Instantiate and train the model (you may need to customize this part)
# model = TransformerModel(input_size=input_size, output_size=output_size)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# for epoch in range(2):
#     optimizer.zero_grad()
#     output_states = model(src_hidden_states.unsqueeze(0))
#     loss = criterion(output_states, tgt_hidden_states.unsqueeze(0))
#     loss.backward(retain_graph=True)  # Retain the graph
#     optimizer.step()
#
#     if epoch % 100 == 0:
#         print(f"Epoch {epoch}, Loss: {loss.item()}")
#
#     torch.cuda.empty_cache()
#
# # Test the trained model
# test_output_states = model(src_hidden_states.unsqueeze(0))
# print("Predicted Hidden States:", test_output_states)


# try using the result as input to the OPT model
# layer = 1
# outputs, generated_text = embeddingsToOPT.OPT_activation_different_layer(test_output_states, layer)
# # print("Generated Text: ", generated_text)
