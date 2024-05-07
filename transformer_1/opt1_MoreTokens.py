import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModel, AutoTokenizer, OPTForCausalLM, AutoModelForCausalLM
from transformers import pipeline
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import embeddingsToOPT
from embeddingsToOPT import CustomLayerWrapper

# load data
# activateTrans1(0,0)

# generator = pipeline('text-generation', model="facebook/opt-350m")
# print(generator("I want to eat the"))


file = "C:\Users\user\Documents\TranslatorGPT\HebrewLLM\resources\He_En_moreTokensData.csv"
df = pd.read_csv(file)
df = df.dropna()
# build models
device = "cuda" if torch.cuda.is_available() else "cpu"
src_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
tgt_model_name = "facebook/opt-350m"
src_model = AutoModelForSeq2SeqLM.from_pretrained(src_model_name).to(device)
tgt_model = AutoModel.from_pretrained(tgt_model_name).to(device)
model = AutoModelForCausalLM.from_pretrained(tgt_model_name).to(device)

# Set input_size and output_size based on the model configurations
input_size = src_model.config.hidden_size
# print(input_size)
output_size = tgt_model.config.hidden_size
output_size1 = model.config.hidden_size
# print(output_size)

# Tokenize sentences
src_tokenizer = AutoTokenizer.from_pretrained(src_model_name)
tgt_tokenizer = AutoTokenizer.from_pretrained(tgt_model_name)

max_length = 10  # Example length, adjust as needed
src_inputs0 = src_tokenizer.batch_encode_plus(df['statement'].tolist(),
                                              return_tensors='pt', truncation=True).to(device)
tgt_inputs0 = tgt_tokenizer.batch_encode_plus(df['translation'].tolist(),
                                              return_tensors='pt', truncation=True).to(device)


# for i in range(3):
#     # Prepare the inputs
#     src_input = src_inputs0['input_ids'][i].unsqueeze(0).to(device)
#     tgt_input = tgt_inputs0['input_ids'][i].unsqueeze(0).to(device)

#     # Forward pass through the source model to get hidden states
#     with torch.no_grad():
#         src_output = src_model.generate(src_input, output_hidden_states=True, return_dict_in_generate=True, max_length=max_length)
#         src_hidden_state = src_output.encoder_hidden_states[-1][0,:,:]  # Last layer's hidden states

#     #inputs = tgt_tokenizer(en_list[i], return_tensors="pt")
#     outputs = model(**tgt_inputs0[i], output_hidden_states=True)

#     # hs = outputs.hidden_states[1]
#     # numpy_array = hs.detach().numpy()
#     # np.save('tensor_data.npy', numpy_array)

#     # Access the probabilities of all tokens
#     token_probs = outputs.logits.softmax(dim=-1)

#     # Get the indices of the top 10 probabilities
#     top_indices = token_probs[0].topk(10).indices
#     for topInds in top_indices:
#         top_generated_texts = [tgt_tokenizer.decode(index.item(), skip_special_tokens=True) for index in topInds]
#         print(top_generated_texts)
# jhdgfjdwkfs=0


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


def train_model(src_model, tgt_model, hidden_state_transformer, optimizer, criterion, num_epochs, device):
    dataSize = int(0.8 * src_inputs0['input_ids'].shape[0])
    # dataSize = 100
    src_hidden_states = torch.empty((dataSize, max_length, input_size))
    tgt_hidden_states = torch.empty((dataSize, max_length, output_size))

    for i in range(dataSize):
        # Prepare the inputs
        src_input = src_inputs0['input_ids'][i].unsqueeze(0).to(device)
        tgt_input = tgt_inputs0['input_ids'][i].unsqueeze(0).to(device)

        # Forward pass through the source model to get hidden states
        with torch.no_grad():
            src_output = src_model.generate(src_input, output_hidden_states=True, return_dict_in_generate=True,
                                            max_length=max_length)
            src_hidden_states[i] = src_output.encoder_hidden_states[-1][0, :, :]  # Last layer's hidden states

        # Forward pass through the target model to get target hidden states
        with torch.no_grad():
            tgt_output = tgt_model(input_ids=tgt_input, output_hidden_states=True)
            tgt_hidden_states[i] = tgt_output.hidden_states[1][0, :, :]  # First layer's hidden states

    torch.save(src_hidden_states, 'srcHS_moreTokens_3000.pth')
    torch.save(tgt_hidden_states, 'tgtHS_moreTokens_3000.pth')
    # src_hidden_states = torch.load('srcHS_moreTokens_3000.pth')
    # tgt_hidden_states = torch.load('tgtHS_moreTokens_3000.pth')
    # Compute the mean along the last dimension
    mean_values = torch.mean(tgt_hidden_states, dim=0)

    # Expand the mean values back to the original shape
    avg_hs = mean_values.unsqueeze(0).expand_as(tgt_hidden_states)
    loss = criterion(avg_hs, tgt_hidden_states)
    print(loss)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predicted_tgt_hidden_states = hidden_state_transformer(src_hidden_states)

        loss = criterion(predicted_tgt_hidden_states, tgt_hidden_states)

        # Backpropagation
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")


def infer_hidden_states(hidden_state_transformer, src_model, src_tokenizer, input_word, device):
    # src_input = src_inputs['input_ids'][2].unsqueeze(0).to(device)

    # # Forward pass through the source model to get hidden states
    # with torch.no_grad():
    #     src_output = src_model.generate(src_input, output_hidden_states=True,
    #                                 return_dict_in_generate=True, max_length=max_length)
    #     src_hidden_states = src_output.encoder_hidden_states[-1]  # Last layer's hidden states
    # Tokenize the input word
    tokenized_input = src_tokenizer.encode_plus(input_word, padding='max_length', max_length=max_length,
                                                return_tensors='pt', truncation=True).to(device)
    print(tokenized_input['attention_mask'].shape)
    # Forward pass through the source model to get hidden states
    with torch.no_grad():
        src_output = src_model.generate(input_ids=tokenized_input['input_ids'].to(device), output_hidden_states=True,
                                        return_dict_in_generate=True, max_length=max_length,
                                        attention_mask=tokenized_input['attention_mask'].to(device))
        src_hidden_states = src_output.encoder_hidden_states[-1]  # [0,:,:]  # Last layer's hidden states

        # print(src_hidden_states.shape)
        # print(src_hidden_states)

    # Forward pass through your transformer model
    with torch.no_grad():
        converted_hidden_states = hidden_state_transformer(src_hidden_states)

    return converted_hidden_states


# Function to load the model
def load_hidden_state_transformer(model_path, input_size, output_size, num_layers, num_heads, dim_feedforward, dropout,
                                  device):
    model = HiddenStateTransformer(input_size, output_size, num_layers, num_heads, dim_feedforward, dropout)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


def main():
    model_save_path = "hidden_state_transformer_moreTokens.pth"

    # Model parameters
    num_layers = 3  # Example, you can tune this
    num_heads = 4  # Example, you can tune this
    dim_feedforward = 512  # Example, you can tune this
    dropout = 0.1  # Example, you can tune this

    # Initialize the transformer model
    hidden_state_transformer = HiddenStateTransformer(input_size, output_size, num_layers, num_heads, dim_feedforward,
                                                      dropout).to(device)
    # hidden_state_transformer.load_state_dict(torch.load(model_save_path))
    # hidden_state_transformer.train()

    # Initialize the loss function and optimizer
    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    # criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(hidden_state_transformer.parameters(), lr=0.001)  # You can tune the learning rate
    # optimizer = torch.optim.SGD(hidden_state_transformer.parameters(), lr=0.1, momentum=0.9)
    num_epochs = 100  # Set the number of epochs for training

    # hidden_state_transformer = load_hidden_state_transformer(model_save_path, input_size, output_size, num_layers,
    #                                                          num_heads, dim_feedforward, dropout, device)
    # Call the training function
    # train_model(src_model, model, hidden_state_transformer, optimizer, criterion, num_epochs, device)
    # # Saving the weights of the HiddenStateTransformer
    # torch.save(hidden_state_transformer.state_dict(), model_save_path)

    # Loading the model
    hidden_state_transformer = load_hidden_state_transformer(model_save_path, input_size, output_size, num_layers,
                                                             num_heads, dim_feedforward, dropout, device)

    # src_hidden_states = torch.load('srcHS_moreTokens_3000.pth')
    # tgt_hidden_states = torch.load('tgtHS_moreTokens_3000.pth')
    # predicted_tgt_hidden_states = hidden_state_transformer(src_hidden_states)
    # loss = criterion(predicted_tgt_hidden_states, tgt_hidden_states)
    # print("Loss on train data is: ", loss.item())

    lll = len(df['statement'].tolist())
    dataSize = int(0.2 * src_inputs0['input_ids'].shape[0])
    # src_hidden_states2 = torch.load('srcHS.pth')
    # new_tgt_hidden_states2 = torch.load('tgtHS.pth')
    # src_hidden_states = torch.load('srcHS_moreTokens_test_3000.pth')
    # tgt_hidden_states = torch.load('tgtHS_moreTokens_test_3000.pth')
    src_hidden_states = torch.empty((dataSize, max_length, 1024))
    tgt_hidden_states = torch.empty((dataSize, max_length, 1024))

    for i in range(dataSize):
        # Prepare the inputs
        src_input = src_inputs0['input_ids'][lll - 1 - i].unsqueeze(0).to(device)
        tgt_input = tgt_inputs0['input_ids'][lll - 1 - i].unsqueeze(0).to(device)

        # Forward pass through the source model to get hidden states
        with torch.no_grad():
            src_output = src_model.generate(src_input, output_hidden_states=True, return_dict_in_generate=True,
                                            max_length=max_length)
            src_hidden_states[i] = src_output.encoder_hidden_states[-1][0, :, :]  # Last layer's hidden states

        # Forward pass through the target model to get target hidden states
        with torch.no_grad():
            tgt_output = tgt_model(input_ids=tgt_input, output_hidden_states=True)
            tgt_hidden_states[i] = tgt_output.hidden_states[1][0, :, :]  # First layer's hidden states
    torch.save(src_hidden_states, 'srcHS_moreTokens_test_3000.pth')
    torch.save(tgt_hidden_states, 'tgtHS_moreTokens_test_3000.pth')
    predicted_tgt_hidden_states = hidden_state_transformer(src_hidden_states)
    loss = criterion(predicted_tgt_hidden_states, tgt_hidden_states)
    print("Loss on test data is: ", loss.item())

    file0 = open("indices.txt", 'a', encoding='UTF8')
    file1 = open("tgt_Probs.txt", 'a', encoding='UTF8')
    file2 = open("predicted_Probs.txt", 'a', encoding='UTF8')
    file3 = open("tgt_Wrds.txt", 'a', encoding='UTF8')
    file4 = open("predicted_Wrds.txt", 'a', encoding='UTF8')
    file5 = open("loss.txt", 'a', encoding='UTF8')
    for i in range(int(0.2 * lll)):
        # Word by word
        input_word = df['statement'].tolist()[lll - 1 - i]  # Replace with your Hebrew word
        print(input_word)
        # Using the function
        converted_hidden_states = infer_hidden_states(hidden_state_transformer, src_model, src_tokenizer, input_word,
                                                      device)
        # file0.write(str(i) + '\n')
        ch = converted_hidden_states.softmax(-1)
        # for j in range(10):
        #     file2.write(str(ch.topk(10).values[0,1,j]) + '\t')
        # file2.write('\n')

        # Continue with your OPT model or any other post-processing steps
        layer = 0
        outputs, generated_text = embeddingsToOPT.OPT_activation_different_layer(converted_hidden_states, layer,
                                                                                 max_length - 1)
        print(generated_text)
        # file4.write(str(generated_text) + '\n')

        tgt_input = tgt_inputs0['input_ids'][lll - 1 - i].unsqueeze(0).to(device)
        indices = torch.where(tgt_input.eq(1))[1]
        first_index = indices[0].item() - 1 if len(indices) > 0 else max_length - 1

        with torch.no_grad():
            tgt_output = tgt_model(input_ids=tgt_input, output_hidden_states=True)
            tgt_hidden_state = tgt_output.hidden_states[1]  # First layer's hidden states
        outputs, generated_text = embeddingsToOPT.OPT_activation_different_layer(tgt_hidden_state, layer, first_index)
        print(generated_text)
        # file3.write(str(generated_text) + '\n')
        # print("By hs: ", generated_text)
        # abc = tgt_hidden_state.softmax(-1)
        # for j in range(10):
        #     file1.write(str(abc.topk(10).values[0,1,j]) + '\t')
        # file1.write('\n')
        # loss = criterion(converted_hidden_states.view(1, -1).unsqueeze(1)[0,0,:], tgt_hidden_state[0,1,:])
        loss = criterion(converted_hidden_states, tgt_hidden_state)
        print(loss)
        # file5.write(str(loss) + '\n')

        input_word = df['statement'].tolist()[i]  # Replace with your Hebrew word
        print(input_word)
        # Using the function
        converted_hidden_states = infer_hidden_states(hidden_state_transformer, src_model, src_tokenizer, input_word,
                                                      device)
        # file0.write(str(i) + '\n')
        ch = converted_hidden_states.softmax(-1)
        # for j in range(10):
        #     file2.write(str(ch.topk(10).values[0,1,j]) + '\t')
        # file2.write('\n')

        # Continue with your OPT model or any other post-processing steps
        layer = 0
        outputs, generated_text = embeddingsToOPT.OPT_activation_different_layer(converted_hidden_states, layer,
                                                                                 max_length - 1)
        print(generated_text)
        # file4.write(str(generated_text) + '\n')

        tgt_input = tgt_inputs0['input_ids'][i].unsqueeze(0).to(device)
        indices = torch.where(tgt_input.eq(1))[1]
        first_index = indices[0].item() - 1 if len(indices) > 0 else max_length
        with torch.no_grad():
            tgt_output = tgt_model(input_ids=tgt_input, output_hidden_states=True)
            tgt_hidden_state = tgt_output.hidden_states[1]  # First layer's hidden states
        outputs, generated_text = embeddingsToOPT.OPT_activation_different_layer(tgt_hidden_state, layer, first_index)
        print(generated_text)
        # file3.write(str(generated_text) + '\n')
        # print("By hs: ", generated_text)
        # abc = tgt_hidden_state.softmax(-1)
        # for j in range(10):
        #     file1.write(str(abc.topk(10).values[0,1,j]) + '\t')
        # file1.write('\n')
        # loss = criterion(converted_hidden_states.view(1, -1).unsqueeze(1)[0,0,:], tgt_hidden_state[0,1,:])
        loss = criterion(converted_hidden_states, tgt_hidden_state)
        print(loss)
        # file5.write(str(loss) + '\n')

    # input_word = "שלום"  # Replace with your Hebrew word

    # Using the function
    # converted_hidden_states = infer_hidden_states(hidden_state_transformer, src_model, src_tokenizer, input_word,
    #                                               device)
    # print(converted_hidden_states)

    # # Continue with your OPT model or any other post-processing steps
    # layer = 0
    # outputs, generated_text = embeddingsToOPT.OPT_activation_different_layer(converted_hidden_states, layer)
    # print("Generated Text: ", generated_text)


if __name__ == '__main__':
    main()