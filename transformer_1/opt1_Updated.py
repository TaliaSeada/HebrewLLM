import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModel, AutoTokenizer, OPTForCausalLM
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import embeddingsToOPT
from embeddingsToOPT import CustomLayerWrapper

# load data
#activateTrans1(0,0)

file = "C:\\Users\\user\\OneDrive\\Desktop\\HebLLM\\pythonProject\\dict0.csv"
df = pd.read_csv(file)
df = df.dropna()
# build models
device = "cuda" if torch.cuda.is_available() else "cpu"
src_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
tgt_model_name = "facebook/opt-350m"
src_model = AutoModelForSeq2SeqLM.from_pretrained(src_model_name).to(device)
tgt_model = AutoModel.from_pretrained(tgt_model_name).to(device)
model = OPTForCausalLM.from_pretrained(tgt_model_name).to(device)

# Set input_size and output_size based on the model configurations
input_size = src_model.config.hidden_size
# print(input_size)
output_size = tgt_model.config.hidden_size
# print(output_size)

# Tokenize sentences
src_tokenizer = AutoTokenizer.from_pretrained(src_model_name)
tgt_tokenizer = AutoTokenizer.from_pretrained(tgt_model_name)


max_length = 2  # Example length, adjust as needed
src_inputs = src_tokenizer.batch_encode_plus(df['statement'].tolist(), padding='max_length', max_length=max_length, return_tensors='pt', truncation=True).to(device)
tgt_inputs = tgt_tokenizer.batch_encode_plus(df['translation'].tolist(), padding='max_length', max_length=max_length, return_tensors='pt', truncation=True).to(device)


# for i in range(tgt_inputs['input_ids'].shape[0]):
#     input_word = df['translation'].tolist()[i]
#     inputs = tgt_tokenizer(input_word, return_tensors="pt")
#     # Using OPTForCausalLM
#     outputs = model(**inputs, output_hidden_states=True)
#     tgt_hidden_states = outputs.hidden_states[0][0,0,:]  # First layer's hidden states
#     print(tgt_hidden_states)
#     generated_text = tgt_tokenizer.decode(torch.argmax(tgt_hidden_states).item(), skip_special_tokens=True)
#     # print(torch.argmax(tgt_hidden_states).item())
#     print("Generated Text: ", generated_text)
#     tgt_hidden_states = outputs.hidden_states[0][0,1,:]  # First layer's hidden states
#     print(tgt_hidden_states)
#     generated_text = tgt_tokenizer.decode(torch.argmax(tgt_hidden_states).item(), skip_special_tokens=True)
#     # print(torch.argmax(tgt_hidden_states).item())
#     print("Generated Text: ", generated_text)

#     # Using AutoModel
#     tgt_input = tgt_inputs['input_ids'][i].unsqueeze(0).to(device)
#     tgt_output = tgt_model(input_ids=tgt_input, output_hidden_states=True)
#     tgt_hidden_states = tgt_output.hidden_states[0][0,0,:]  # First layer's hidden states
#     print(tgt_hidden_states)
#     generated_text = tgt_tokenizer.decode(torch.argmax(tgt_hidden_states).item(), skip_special_tokens=True)
#     print("Generated Text: ", generated_text)
#     tgt_hidden_states = tgt_output.hidden_states[0][0,1,:]  # First layer's hidden states
#     print(tgt_hidden_states)
#     generated_text = tgt_tokenizer.decode(torch.argmax(tgt_hidden_states).item(), skip_special_tokens=True)
#     print("Generated Text: ", generated_text)
#     # token_ids = outputs.logits.argmax(-1)
#     # generated_text = tgt_tokenizer.decode(token_ids[0], skip_special_tokens=True)
#     # print("Generated Text: ", generated_text)



# Transformer
class HiddenStateTransformer(nn.Module):
    def __init__(self, input_size, output_size, num_layers, num_heads, dim_feedforward, dropout=0.1):
        super(HiddenStateTransformer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        

        # Transformer Encoder Layer
        encoder_layers = TransformerEncoderLayer(d_model=input_size, nhead=num_heads,
                                                 dim_feedforward=dim_feedforward, dropout=dropout)
        encoder_layers.self_attn.batch_first=True
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # Linear layer to map to the target hidden size
        self.fc = nn.Linear(1024, 512)

    def forward(self, src):
        # src shape: (seq_length, batch_size, input_size)
        encoded = self.transformer_encoder(src)
        # encoded shape: (seq_length, batch_size, input_size)
        # Apply ReLU activation function
        # activated = F.relu(encoded)
        output = self.fc(encoded)
        # output shape: (seq_length, batch_size, output_size)
        return output


def train_model(src_model, tgt_model, hidden_state_transformer, src_inputs, tgt_inputs, optimizer, criterion, num_epochs, device):
    src_hidden_states = torch.empty((src_inputs['input_ids'].shape[0],2,1024))
    tgt_hidden_states = torch.empty((src_inputs['input_ids'].shape[0],2,1024))
    dataSize = int(0.8*src_inputs['input_ids'].shape[0])
    for i in range(dataSize):
        # Prepare the inputs
        src_input = src_inputs['input_ids'][i].unsqueeze(0).to(device)
        tgt_input = tgt_inputs['input_ids'][i].unsqueeze(0).to(device)

        # Forward pass through the source model to get hidden states
        with torch.no_grad():
            src_output = src_model.generate(src_input, output_hidden_states=True, return_dict_in_generate=True, max_length=max_length)
            src_hidden_states[i] = src_output.encoder_hidden_states[-1][0,:,:]  # Last layer's hidden states
            #print(src_hidden_states[i])
        # print(src_output)
        # print(src_output.encoder_hidden_states[1])
        # print(src_hidden_states[i])

        # Forward pass through the target model to get target hidden states
        with torch.no_grad():
            tgt_output = tgt_model(input_ids=tgt_input, output_hidden_states=True)
            tgt_hidden_states[i] = tgt_output.hidden_states[1][0,:,:]  # First layer's hidden states
            # src_hidden_states[i][0][0] = tgt_hidden_states[i][0][0]
            
    # new_tgt_hidden_states = torch.empty((src_inputs['input_ids'].shape[0],1,1,1024))
    new_tgt_hidden_states = tgt_hidden_states[:, 1, :].unsqueeze(1).to(device)

    for epoch in range(num_epochs):
        # tgt_hidden_states = torch.empty((src_inputs['input_ids'].shape[0],1,2,1024))
        # predicted_tgt_hidden_states = torch.empty((src_inputs['input_ids'].shape[0],1,2,512)).to(device)
        
        # for i in range(src_inputs['input_ids'].shape[0]):
        #     # Prepare the inputs
        #     src_input = src_inputs['input_ids'][i].unsqueeze(0).to(device)
        #     tgt_input = tgt_inputs['input_ids'][i].unsqueeze(0).to(device)

        #     # Forward pass through the source model to get hidden states
        #     with torch.no_grad():
        #         src_output = src_model.generate(src_input, output_hidden_states=True,
        #                                          return_dict_in_generate=True, max_length=max_length)
        #         # src_output = src_model(input_ids=src_input, output_hidden_states=True)
        #         src_hidden_states = src_output.encoder_hidden_states[-1]  # Last layer's hidden states
        #         # Forward pass through your transformer model
        #     predicted_tgt_hidden_states[i] = hidden_state_transformer(src_hidden_states)

        #     # Forward pass through the target model to get target hidden states
        #     with torch.no_grad():
        #         # print(torch.max(tgt_input).item())
        #         # gt = tgt_tokenizer.decode(torch.max(tgt_input).item(), skip_special_tokens=True)
        #         # print(gt)
        #         tgt_output = tgt_model(input_ids=tgt_input, output_hidden_states=True)
        #     tgt_hidden_states[i] = tgt_output.hidden_states[0]  # First layer's hidden states
        #         # generated_text = tgt_tokenizer.decode(torch.argmax(tgt_hidden_states).item(), skip_special_tokens=True)
        #         # print(torch.argmax(tgt_hidden_states).item())
        #         # print("Generated Text: ", generated_text)
        optimizer.zero_grad()
        # for i in range(src_inputs['input_ids'].shape[0]):
        #     predicted_tgt_hidden_states[i] = hidden_state_transformer(src_hidden_states[i])
        #     # print(src_hidden_states[i])
        #     # print(predicted_tgt_hidden_states[i])
        #     # print(tgt_hidden_states[i])
        predicted_tgt_hidden_states = hidden_state_transformer(src_hidden_states)
        new_predicted_tgt_hidden_states = predicted_tgt_hidden_states.view(src_inputs['input_ids'].shape[0], 1, -1)

        # if epoch==num_epochs-1:
        #     file2 = "C:\\Users\\user\\OneDrive\\Desktop\\HebLLM\\pythonProject\\loss.txt"
        #     with open(file2, 'a', encoding='UTF8') as f:
        #         for l in range(5):
        #             for p in range(1024):
        #                 f.write(str(new_predicted_tgt_hidden_states[l,0,0,p]) + "\t" + str(new_tgt_hidden_states[l,0,0,p]) + "\n")


        loss = criterion(new_predicted_tgt_hidden_states, new_tgt_hidden_states)

        # Backpropagation
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")


def infer_hidden_states(hidden_state_transformer, src_model, src_tokenizer, input_word, device):
    # src_input = src_inputs['input_ids'][2].unsqueeze(0).to(device)

    # # Forward pass through the source model to get hidden states
    # with torch.no_grad():
    #     src_output = src_model.generate(src_input, output_hidden_states=True,
    #                                 return_dict_in_generate=True, max_length=max_length)
    #     src_hidden_states = src_output.encoder_hidden_states[-1]  # Last layer's hidden states
    #Tokenize the input word
    tokenized_input = src_tokenizer.encode(input_word, return_tensors="pt", truncation=True, padding=True).to(device)

    #Forward pass through the source model to get hidden states
    with torch.no_grad():
        src_output = src_model.generate(tokenized_input, output_hidden_states=True, return_dict_in_generate=True, max_length=max_length)
        src_hidden_states = src_output.encoder_hidden_states[-1]  # Last layer's hidden states
        # print(src_hidden_states.shape)
        # print(src_hidden_states)

    # Forward pass through your transformer model
    with torch.no_grad():
        converted_hidden_states = hidden_state_transformer(src_hidden_states)
    # converted_hidden_states = hidden_state_transformer(src_hidden_states)

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
    # hidden_state_transformer.load_state_dict(torch.load(model_save_path))
    # hidden_state_transformer.train()

    # Initialize the loss function and optimizer
    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    # criterion = nn.KLDivLoss(reduction="batchmean")
    # optimizer = torch.optim.Adam(hidden_state_transformer.parameters(), lr=0.05)  # You can tune the learning rate
    optimizer = torch.optim.SGD(hidden_state_transformer.parameters(), lr=0.1, momentum=0.9)
    num_epochs = 300  # Set the number of epochs for training

    # hidden_state_transformer = load_hidden_state_transformer(model_save_path, input_size, output_size, num_layers,
    #                                                          num_heads, dim_feedforward, dropout, device)
    # Call the training function
    # train_model(src_model, tgt_model, hidden_state_transformer, src_inputs, tgt_inputs, optimizer, criterion, num_epochs, device)
    # # Saving the weights of the HiddenStateTransformer
    # torch.save(hidden_state_transformer.state_dict(), model_save_path)

    # Loading the model
    hidden_state_transformer = load_hidden_state_transformer(model_save_path, input_size, output_size, num_layers,
                                                             num_heads, dim_feedforward, dropout, device)
    # Example Hebrew word

    input_word0 = df['statement'].tolist()[1]   # Replace with your Hebrew word
    converted_hidden_states0 = infer_hidden_states(hidden_state_transformer, src_model, src_tokenizer, input_word0,
                                                  device)
    # tokenized_input = src_tokenizer.encode(input_word, return_tensors="pt", truncation=True, padding=True).to(device)
    tgt_input = tgt_inputs['input_ids'][1].unsqueeze(0).to(device)
    with torch.no_grad():
        tgt_output = tgt_model(input_ids=tgt_input, output_hidden_states=True)
        tgt_hidden_states = tgt_output.hidden_states[1]  # First layer's hidden states
    lll = len(df['statement'].tolist())
    
    # for i in range(lll-1):
    #     input_word = df['statement'].tolist()[i]   # Replace with your Hebrew word
    #     converted_hidden_states = infer_hidden_states(hidden_state_transformer, src_model, src_tokenizer, input_word,device)
    #     if converted_hidden_states.shape != converted_hidden_states0.shape:
    #         continue
    #     tgt_input = tgt_inputs['input_ids'][i].unsqueeze(0).to(device)
    #     with torch.no_grad():
    #         tgt_output = tgt_model(input_ids=tgt_input, output_hidden_states=True)
    #         tgt_hidden_state = tgt_output.hidden_states[1]  # First layer's hidden states
    #     for j in range(lll-i-1):
    #         input_word2 = df['statement'].tolist()[i+j+1]   # Replace with your Hebrew word
    #         converted_hidden_states2 = infer_hidden_states(hidden_state_transformer, src_model, src_tokenizer, input_word2,device)
    #         if converted_hidden_states2.shape != converted_hidden_states0.shape:
    #             continue
    #         criterion1 = nn.MSELoss()
    #         loss = criterion1(converted_hidden_states, converted_hidden_states2)
    #         tgt_input2 = tgt_inputs['input_ids'][i+j+1].unsqueeze(0).to(device)
    #         with torch.no_grad():
    #             tgt_output2 = tgt_model(input_ids=tgt_input2, output_hidden_states=True)
    #             tgt_hidden_state2 = tgt_output2.hidden_states[1]  # First layer's hidden states
    #         criterion2 = nn.MSELoss()
    #         loss2 = criterion2(tgt_hidden_state[0,1,:], tgt_hidden_state2[0,1,:])
    #         print(str(loss) + "\t" + str(loss2))


    # Calculate all the converted hidden states together    
    # src_hidden_states = torch.empty((src_inputs['input_ids'].shape[0],2,1024))
    # for i in range(lll):
    #     # Prepare the inputs
    #     src_input = src_inputs['input_ids'][i].unsqueeze(0).to(device)

    #     # Forward pass through the source model to get hidden states
    #     with torch.no_grad():
    #         src_output = src_model.generate(src_input, output_hidden_states=True, return_dict_in_generate=True, max_length=max_length)
    #         src_hidden_states[i] = src_output.encoder_hidden_states[-1][0,:,:]  # Last layer's hidden states
    # with torch.no_grad():
    #     converted_hidden_states = hidden_state_transformer(src_hidden_states)
    
    # hbWrds = ['אור','גינה','ספינה','שיר','מראה','נעל','תיק']
    # enWrds = ['Light','Garden','Ship','Song','Appearance','Shoe','Bag']
    # tgt = tgt_tokenizer.batch_encode_plus(enWrds, padding='max_length', max_length=max_length, return_tensors='pt', truncation=True).to(device)
    # for i in range(7):
    #     input_word = hbWrds[i]
        
    #     # Using the function
    #     converted_hidden_states = infer_hidden_states(hidden_state_transformer, src_model, src_tokenizer, input_word,device)
    #     if converted_hidden_states.shape != converted_hidden_states0.shape:
    #         continue
    #     print(i)
    #     # print(converted_hidden_states)
    #     new_converted_hidden_states = tgt_hidden_states
    #     new_converted_hidden_states[0,1,:] = converted_hidden_states.view(1, -1).unsqueeze(1)
    #     ch=new_converted_hidden_states.softmax(-1)
    #     print(ch.topk(10).values)


    #     # Continue with your OPT model or any other post-processing steps
    #     layer = 0
    #     outputs, generated_text = embeddingsToOPT.OPT_activation_different_layer(new_converted_hidden_states, layer)
    #     print("By transformer: ", generated_text)

    #     inputs = tgt_tokenizer(enWrds[i], return_tensors="pt")
    #     outputs0 = model(**inputs, output_hidden_states=True)
    #     token_ids = outputs0.logits.argmax(-1)
    #     generated_text = tgt_tokenizer.decode(token_ids[0], skip_special_tokens=True)
    #     print("By word: ", generated_text)

    #     tgt_input = tgt['input_ids'][i].unsqueeze(0).to(device)
    #     with torch.no_grad():
    #         tgt_output = tgt_model(input_ids=tgt_input, output_hidden_states=True)
    #         tgt_hidden_state = tgt_output.hidden_states[1]  # First layer's hidden states
    #     # if tgt_hidden_state.shape != converted_hidden_states0.shape:
    #     #     continue
    #     # print(tgt_hidden_state)
    #     outputs, generated_text = embeddingsToOPT.OPT_activation_different_layer(tgt_hidden_state, layer)
    #     print("By hs: ", generated_text)
    #     abc = tgt_hidden_state.softmax(-1)
    #     print(abc.topk(10).values)
        
    #     loss = criterion(converted_hidden_states.view(1, -1).unsqueeze(1)[0,0,:], tgt_hidden_state[0,1,:])
    #     print(loss)
    


       
    file0 = open("indices.txt", 'a', encoding='UTF8')
    file1 = open("tgt_Probs.txt", 'a', encoding='UTF8')
    file2 = open("predicted_Probs.txt", 'a', encoding='UTF8')
    file3 = open("tgt_Wrds.txt", 'a', encoding='UTF8')
    file4 = open("predicted_Wrds.txt", 'a', encoding='UTF8')
    file5 = open("loss.txt", 'a', encoding='UTF8')
    for i in range(int(0.2*lll)):
        #Word by word
        input_word = df['statement'].tolist()[lll-1-i]   # Replace with your Hebrew word
        
        # converted_hidden_states = infer_hidden_states(hidden_state_transformer, src_model, src_tokenizer, input_word,device)
        # if converted_hidden_states.shape != converted_hidden_states0.shape:
        #     continue
        # tokenized_input = src_tokenizer.encode(input_word, return_tensors="pt", truncation=True, padding=True).to(device)
        
        # #Forward pass through the source model to get hidden states
        # with torch.no_grad():
        #     tgt_output = model.generate(tokenized_input, output_hidden_states=True, return_dict_in_generate=True, max_length=max_length)
        #     tgt_hidden_states = tgt_output.encoder_hidden_states[-1]  # Last layer's hidden states
        
        # outputs, generated_text = embeddingsToOPT.OPT_activation_different_layer(tgt_hidden_states, 0)
        # print("Generated Text: ", generated_text)
        
        # Using the function
        converted_hidden_states = infer_hidden_states(hidden_state_transformer, src_model, src_tokenizer, input_word,device)
        if converted_hidden_states.shape != converted_hidden_states0.shape:
            continue
        file0.write(str(i) + '\n')
        # print(df['statement'].tolist()[lll-1-i])
        # print(converted_hidden_states)
        new_converted_hidden_states = tgt_hidden_states
        new_converted_hidden_states[0,1,:] = converted_hidden_states.view(1, -1).unsqueeze(1)
        ch=new_converted_hidden_states.softmax(-1)
        for j in range(10):
            file2.write(str(ch.topk(10).values[0,1,j]) + '\t')
        file2.write('\n')
        # print(tgt_hidden_states[0,:,:,:])
        # print(converted_hidden_states.view(1, 1, -1))
        # print(new_converted_hidden_states)


        # Continue with your OPT model or any other post-processing steps
        layer = 0
        # generated_text = tgt_tokenizer.decode(torch.argmax(converted_hidden_states).item(), skip_special_tokens=True)
        outputs, generated_text = embeddingsToOPT.OPT_activation_different_layer(new_converted_hidden_states, layer)
        file4.write(str(generated_text) + '\n')
        # print("By transformer: ", generated_text)

        # inputs = tgt_tokenizer(df['translation'].tolist()[i], return_tensors="pt")
        # outputs0 = model(**inputs, output_hidden_states=True)
        # token_ids = outputs0.logits.argmax(-1)
        # generated_text = tgt_tokenizer.decode(token_ids[0], skip_special_tokens=True)
        # print("By word: ", generated_text)

        tgt_input = tgt_inputs['input_ids'][lll-1-i].unsqueeze(0).to(device)
        with torch.no_grad():
            tgt_output = tgt_model(input_ids=tgt_input, output_hidden_states=True)
            tgt_hidden_state = tgt_output.hidden_states[1]  # First layer's hidden states
        # if tgt_hidden_state.shape != converted_hidden_states0.shape:
        #     continue
        # print(tgt_hidden_state)
        outputs, generated_text = embeddingsToOPT.OPT_activation_different_layer(tgt_hidden_state, layer)
        file3.write(str(generated_text) + '\n')
        # print("By hs: ", generated_text)
        abc = tgt_hidden_state.softmax(-1)
        for j in range(10):
            file1.write(str(abc.topk(10).values[0,1,j]) + '\t')
        file1.write('\n')
        # print(abc.topk(10).values)
        
        loss = criterion(converted_hidden_states.view(1, -1).unsqueeze(1)[0,0,:], tgt_hidden_state[0,1,:])
        file5.write(str(loss) + '\n')
        # print(loss)


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
