import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from transformers import AutoTokenizer, AutoModel, MarianTokenizer,MarianMTModel, AutoTokenizer, OPTForCausalLM
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from data.dataManipulation import pad, pad_and_mask
from model.HiddenStateTransformer import HiddenStateTransformer,HiddenStateTransformer2, train_model, test_model
from generalTransformer import CustomLayerWrapper, CustomLayerWrapper2
import torch.nn.functional as F
import math


device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_with_logits(model, tokenizer, ids, attention_mask):
    # Prepare decoder_input_ids, starting with the <pad> token
    
    # print(f"ids = {ids}")
    print(f"translated tokens  = {tokenizer.convert_ids_to_tokens(ids)}")
    
    # print(f"inputs.input_ids.size(0) = {inputs.input_ids.size(0)}")
    decoder_input_ids = torch.full(
        (ids.size(0), 15), tokenizer.pad_token_id, dtype=torch.long, device=ids.device
    )
    # print(f"decoder_input_ids = {decoder_input_ids},\nShape = {decoder_input_ids.shape}")
    
    # Forward pass to get the logits
    outputs = model(
        input_ids=ids, 
        attention_mask=attention_mask, 
        decoder_input_ids=decoder_input_ids,
        output_hidden_states=True
    )
    logits = outputs.logits
    logits = logits.requires_grad_()
    
    # Apply softmax to get probabilities
    probabilities = F.log_softmax(logits, dim=-1)
    
    return logits, probabilities

class CombinedModel(nn.Module):
    def __init__(self, transformer1, transformer2, llm):
        super(CombinedModel, self).__init__()
        self.transformer1 = transformer1
        self.llm = llm
        self.transformer2 = transformer2
        
        # Freeze LLM parameters
        for param in self.llm.parameters():
            param.requires_grad = False
    
    def forward(self, text, tokenizer1, translator1, llm_tokenizer, tokenizer2, translator2):
        
        with torch.no_grad():
            # Get the final embedding of translator1 for the text input (language1)  
            x, _ = hebrew_to_input(text, tokenizer1, translator1)
            # print(x)
        
        # Transform it to the llm initial embeddings for the sentence in language2 
        x = self.transformer1(x)
        
        # Ensure LLM does not compute gradients
        with torch.no_grad():
            
            # Trick llm by giving it a dummy that contain the desired number of tokens (In out case 15)
            # and replace the first layer as it got other word embedding.
            inputs = llm_tokenizer(" " * 14, return_tensors="pt")
            
            # Inject our initial embeddings to the first layer of the llm
            self.inject_layer(model=self.llm, layer_hs=x, layer_num=1, name="decoder")
            
            # Calculates the final embeddings
            outputs = self.llm(**inputs, output_hidden_states=True)
            
            # Lastly, get the final embeddings of the llm for our injected input
            llm_last_hidden_state = outputs.hidden_states[-1]
            
        # TODO - Ensure its 15 Tokens as well
        
        # Transform it (embeddings of output sentence in language 2) to the initial embeddings of translator2
        x = self.transformer2(llm_last_hidden_state)
        # print(f"x.shape = {x.shape}")
        
        with torch.no_grad():
            # Do the same trick as before
            inputs = tokenizer2("a " * 14, return_tensors="pt")
            
            print(f"inputs.input_ids = {inputs.input_ids}")
            
            self.inject_layer(model=translator2, layer_hs=x, layer_num=1, name="encoder")

            # Ensure the attention mask is correctly shaped
            attention_mask = torch.ones((1, 15)).to(x.device)
            inputs['attention_mask'] = attention_mask

            print(f"attention_mask = {inputs['attention_mask']}\nShape = {attention_mask.shape}")
            ids = translator2.generate(inputs.input_ids)[0]
            
            # return the probability for each word in the dictionary for each vector in the final embeddings of translator2
            l, p = generate_with_logits(translator2,En_He_tokenizer,ids,attention_mask)
            
            
            # TODO - why  the logits stay the same
            
            # print(attention_mask,l.shape,p.shape)
            # print(p)
        return l , p

    def test():
        pass
    
    def inject_layer(self, model, layer_hs, layer_num, name="decoder"):
        if name == "decoder":
            original_layer = model.base_model.decoder.layers[layer_num]
            wrapped_layer = CustomLayerWrapper(original_layer, layer_hs)
        else:
            original_layer = model.model.encoder.layers[layer_num]
            wrapped_layer = CustomLayerWrapper2(original_layer, layer_hs)
        
        # Wrap the layer inside the custom wrapper
        
        # Replace the layer with the wrapped layer
        if name == "decoder":
            model.base_model.decoder.layers[layer_num] = wrapped_layer
        else:
            model.model.encoder.layers[layer_num] = wrapped_layer
    
    
def hebrew_to_input(h_text, hebrew_translator_tokenizer,hebrew_translator_model):
    # Translator
    inputs = hebrew_translator_tokenizer(h_text, return_tensors="pt")
    
    # Encode the source text
    generated_ids = hebrew_translator_model.generate(inputs.input_ids)
    
    # print(f"Hebrew input ids = {len(generated_ids[0])}")

    clean_token_num = len(generated_ids[0]) -2
    
    # for t in generated_ids:
    #     print(f"translated token1 {t} = {hebrew_translator_tokenizer.convert_ids_to_tokens(t)}")

    # Append hidden states
    translator_outputs = hebrew_translator_model(input_ids=inputs.input_ids, decoder_input_ids=generated_ids, output_hidden_states=True)
    
    # Extract the last hidden state from translator
    translator_last_hidden_state = translator_outputs.decoder_hidden_states[-1]

    data = [(pad(translator_last_hidden_state),)]
    data_padded, labels_padded, data_masks, labels_masks = pad_and_mask(data, False)

    # print(f"data_padded = {data_padded}")
    return data_padded, clean_token_num


def translate_to_english(model, tokenizer, sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model.generate(**inputs)
    translated_sentence = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_sentence

def translate_to_hebrew(model, tokenizer, sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model.generate(**inputs)
    return translated_tokens


def train_combined_model(dataset_path, stop_index, model: CombinedModel, He_En_model, En_He_model, He_En_tokenizer, En_He_tokenizer, criterion, optimizer,epochs):

    df = pd.read_csv(dataset_path)

    # Train the model
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        train_loss = 0
        
        for index, row in df.iterrows():
            if index > stop_index:
                break
            hebrew_sentence = row[0]
            target_hebrew_sentence  = row[0] + " " + row[1]
            
            
            # ==== TODO - Black box -> New Hebrew Sentence outputs ====
            
            # Outputs the first layer of the En_He_translator using transformer1
            logits, distribution = model(
                hebrew_sentence,
                He_En_tokenizer,
                He_En_model,
                llm_tokenizer,
                En_He_tokenizer,
                En_He_model)

            # print(f"hebrew_ids = {generated_output[0]}")
            
            # # Translation for us
            # hebrew_sentence = En_He_tokenizer.decode(generated_output[0], skip_special_tokens=True)
            # print(f"generated_text = {hebrew_sentence}")
            
            # ==== calc loss =====
            
            # Get the tokens for the target sentence
            with En_He_tokenizer.as_target_tokenizer():
                target_ids = En_He_tokenizer.encode(target_hebrew_sentence, return_tensors="pt")
            
            # print(target_ids[:,:15].squeeze(0).shape)
            
            # TODO - Pad the tensors to 15 so their size will match
                        
            # Calculate the loss
            print(f"logits = {logits.squeeze(0)[1:2,:]}")
            loss = criterion(logits.squeeze(0)[1:2,:], target_ids.squeeze(0)[:1])
        
            # Back Propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # print(f"Loss: {loss}")

        train_loss /= stop_index
        print(f"Epoch {epoch}, Loss = {train_loss}")



def my_cross_entropy(dist, target):
    # return sum([(math.log(dist[0,index,token_id])) if index < 15 else 0 for index, token_id in enumerate(target.squeeze(0))])
    # print(target.view(-1))
    return F.nll_loss(dist, target)
    

def save_model(model, to_path):
    joblib.dump(model, to_path)
    
# Hebrew to english translator
He_En_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"

He_En_tokenizer = MarianTokenizer.from_pretrained(He_En_model_name)
He_En_translator_model = MarianMTModel.from_pretrained(He_En_model_name)



# Transformer 1
t1 = joblib.load('transformer_1/orel/pretrainedModels/models/10Tokens/general_model.pkl')


# LLM model
llm_model_name = "facebook/opt-350m"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm = OPTForCausalLM.from_pretrained(llm_model_name).to(device)

# Transformer 2
# t2 = joblib.load('transformer_2/model_name.pkl')
t2 = HiddenStateTransformer2(input_size=512,output_size=512, num_layers=1, num_heads=2, dim_feedforward=256, dropout=0.15)


# English to Hebrew translator
En_He_model_name = "Helsinki-NLP/opus-mt-en-he"
En_He_tokenizer = MarianTokenizer.from_pretrained(En_He_model_name)
En_He_translator_model = MarianMTModel.from_pretrained(En_He_model_name)


combined_model = CombinedModel(transformer1=t1,transformer2=t2,llm=llm)

# for name, param in combined_model.named_parameters():
#     if param.requires_grad:
#         print(f'{name} requires grad')


# optimizer = optim.Adam(combined_model.parameters(), lr=0.001)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, combined_model.parameters()), lr=1e-4)

criterion = nn.CrossEntropyLoss()
# criterion = my_cross_entropy

train_combined_model('wikipedia_data.csv',
                     30,
                     combined_model,
                     He_En_translator_model,
                     En_He_translator_model,
                     He_En_tokenizer,
                     En_He_tokenizer,
                     criterion,
                     optimizer,
                     3)

num = 1
save_model(combined_model, f'transformer_1/orel/pretrainedModels/models/combined/model_wiki_{num}.pkl')

