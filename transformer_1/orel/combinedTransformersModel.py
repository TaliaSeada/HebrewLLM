import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from transformers import AutoTokenizer, AutoModel, MarianTokenizer,MarianMTModel, AutoTokenizer, OPTForCausalLM
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from data.dataManipulation import pad, pad_and_mask
from model.HiddenStateTransformer import HiddenStateTransformer, train_model, test_model
from generalTransformer import CustomLayerWrapper


device = "cuda" if torch.cuda.is_available() else "cpu"


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
            # Get the final emmbeding of translator1 for the text input (language1)  
            x = hebrew_to_input(text, tokenizer1, translator1)
        
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
        
        with torch.no_grad():
            # Do the same trick as before
            inputs = tokenizer2(" " * 14, return_tensors="pt")
            
            self.inject_layer(model=self.llm, layer_hs=x, layer_num=1, name="encoder")
            
            # Translate back to language1
            outputs = translator2(**inputs, output_hidden_states=True)
            
            # return the probabily for each word in the dictionary for each vector in the final embeddings of translator2
        
        return x

    def test():
        pass
    
    def inject_layer(self, model, layer_hs, layer_num, name="decoder"):
        if name == "decoder":
            original_layer = model.base_model.decoder.layers[layer_num]
        else:
            original_layer = model.model.encoder.layers[layer_num]
        
        # Wrap the layer inside the custom wrapper
        wrapped_layer = CustomLayerWrapper(original_layer, layer_hs)
        
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
    
    print(f"Hebrew input ids = {len(generated_ids[0])}")

    clean_token_num = len(generated_ids[0]) -2
    
    for t in generated_ids:
        print(f"translated token {t} = {hebrew_translator_tokenizer.convert_ids_to_tokens(t)}")

    # Append hidden states
    translator_outputs = hebrew_translator_model(input_ids=inputs.input_ids, decoder_input_ids=generated_ids, output_hidden_states=True)
    
    # Extract the last hidden state from translator
    translator_last_hidden_state = translator_outputs.decoder_hidden_states[-1]

    data = [(pad(translator_last_hidden_state),)]
    data_padded, labels_padded, data_masks, labels_masks = pad_and_mask(data, False)

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


def train_combined_model(model, He_En_model, En_He_model, He_En_tokenizer, En_He_tokenizer, criterion, optimizer,epochs):

    df = pd.read_csv('wikipedia_data.csv')

    # Train the model
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        train_loss = 0
        
        for index, row in df.iterrows():
            hebrew_sentence = row[0]
            target_hebrew_sentence  = row[0] + " " + row[1]
            
            
            # ==== TODO - Black box -> New Hebrew Sentence outputs ====
            
            # Outputs the first layer of the En_He_translator 
            En_He_translator_first_layer = model(hebrew_to_input(hebrew_sentence, He_En_tokenizer, He_En_model))
            
            # Replace the translator first layer with our layer
            original_layer = En_He_model.model.encoder.layers[1]
            wrapped_layer = CustomLayerWrapper(original_layer, En_He_translator_first_layer)
            En_He_model.model.encoder.layers[1] = wrapped_layer

            # TODO depends on the input size
            
            # Get the decoder ids after the injection of the first layer
            inputs = En_He_tokenizer(" " * 14, return_tensors="pt")
            decoder_start_token_id = En_He_tokenizer.pad_token_id
            decoder_input_ids = torch.full((inputs.input_ids.size(0), 1), decoder_start_token_id, dtype=torch.long).to(inputs.input_ids.device)

            # Custom model call
            outputs = En_He_model(input_ids=inputs.input_ids, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
    
            # ==== TODO - Black box -> New Hebrew Sentence outputs ====
    
            # Lets see the hebrew sentense we got (for us - not necessary for the calc)
            generated_ids = En_He_model.generate(inputs.input_ids)

            generated_text = En_He_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
            print(f"generated_text = {generated_text}")
            
            # ==== calc loss =====
            target_inputs = En_He_tokenizer(target_hebrew_sentence, return_tensors="pt", padding=True, truncation=True)
            
            logits = outputs.logits


            # Ensure both token sequences are the same length by padding/truncating
            max_length = max(logits.shape[1], target_inputs.input_ids.shape[1])
            if logits.shape[1] < max_length:
                padding = torch.full((logits.shape[0], max_length - logits.shape[1], logits.shape[2]), En_He_tokenizer.pad_token_id).to(logits.device)
                logits = torch.cat([logits, padding], dim=1)
            if target_inputs.input_ids.shape[1] < max_length:
                padding = torch.full((target_inputs.input_ids.shape[0], max_length - target_inputs.input_ids.shape[1]), En_He_tokenizer.pad_token_id).to(target_inputs.input_ids.device)
                target_inputs.input_ids = torch.cat([target_inputs.input_ids, padding], dim=1)

            # Reshape logits and target to calculate loss
            logits = logits.view(-1, logits.size(-1))
            target_ids = target_inputs.input_ids.view(-1)

            # Calculate the loss
            loss = criterion(logits, target_ids)
            
            train_loss += loss.item()
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item()}")

        train_loss /= len(df.iterrows())


# # Hebrew to english translator
# He_En_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"

# He_En_tokenizer = MarianTokenizer.from_pretrained(He_En_model_name)
# He_En_translator_model = MarianMTModel.from_pretrained(He_En_model_name)



# # Transformer 1
# t1 = joblib.load('transformer_1/orel/pretrainedModels/models/10Tokens/general_model.pkl')


# # LLM model
# llm_model_name = "facebook/opt-350m"
# llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
# llm = OPTForCausalLM.from_pretrained(llm_model_name).to(device)

# # Transformer 2
# # t2 = joblib.load('transformer_2/model_name.pkl')
# t2 = HiddenStateTransformer(num_layers=1, num_heads=2, dim_feedforward=256, dropout=0.15)


# # English to Hebrew translator
# En_He_model_name = "Helsinki-NLP/opus-mt-en-he"
# En_He_tokenizer = MarianTokenizer.from_pretrained(En_He_model_name)
# En_He_translator_model = MarianMTModel.from_pretrained(En_He_model_name)


# combined_model = CombinedModel(transformer1=t1,transformer2=t2,llm=llm)
# optimizer = optim.Adam(combined_model.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss()

# train_combined_model(combined_model,
#                      He_En_translator_model,
#                      En_He_translator_model,
#                      He_En_tokenizer,
#                      En_He_tokenizer,
#                      criterion,
#                      optimizer,
#                      5)

