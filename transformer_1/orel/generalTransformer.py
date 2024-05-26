import TransGeneralEmbeddingToOPT as ge
import torch.nn as nn
import torch.optim as optim
from transformers import MarianTokenizer,MarianMTModel, AutoTokenizer, OPTForCausalLM,AutoModel
import torch
import joblib


device = "cuda" if torch.cuda.is_available() else "cpu"


# Hebrew to english translator
translator_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"

translator_tokenizer = MarianTokenizer.from_pretrained(translator_model_name)
translator_model = MarianMTModel.from_pretrained(translator_model_name)

# OPT model
OPT_model_name = "facebook/opt-350m"
OPT_tokenizer = AutoTokenizer.from_pretrained(OPT_model_name)
opt_model = OPTForCausalLM.from_pretrained(OPT_model_name).to(device)


def your_input_modification(hidden_states):
    # loaded_array = np.load('tensor_data.npy')
    # tensor_data = torch.tensor(loaded_array)
    # modified_states = tensor_data

    modified_states = hidden_states
    return modified_states

class CustomLayerWrapper(nn.Module):
    def __init__(self, layer, hidden_states):
        super().__init__()
        self.layer = layer
        self.hs = hidden_states #transformer's result


    def forward(self, hidden_states, attention_mask=None, layer_head_mask=None,
                past_key_value=None, output_attentions=None, use_cache=None):
        # Apply modifications to hidden_states here
        modified_hidden_states = your_input_modification(self.hs)

        # Pass modified_hidden_states to the original layer
        return self.layer(modified_hidden_states, attention_mask, layer_head_mask,
                          past_key_value, output_attentions, use_cache)


def generate_test_inputs(curr_song):
    # Translator
    inputs = translator_tokenizer(curr_song, return_tensors="pt")
    
    # Encode the source text
    generated_ids = translator_model.generate(inputs.input_ids)

    # Translation to english
    english_text = translator_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print(english_text[:-1])
    english_text = english_text[:-1]
    
    # OPT
    opt_inputs = OPT_tokenizer(english_text, return_tensors="pt")
            
                
    # Append hidden states
    translator_outputs = translator_model(input_ids=inputs.input_ids, decoder_input_ids=generated_ids, output_hidden_states=True)
    
    # # decoder_input_ids = opt_inputs.input_ids[:, 1:]  # This removes the first token, usually a start token
    # decoder_input_ids = torch.cat([opt_inputs.input_ids, torch.tensor([[translator_tokenizer.eos_token_id]]).to(opt_inputs.input_ids.device)], dim=1)  # Append EOS token

    # opt_outputs = opt_model(input_ids=opt_inputs.input_ids, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
    opt_outputs = opt_model(input_ids=opt_inputs.input_ids, output_hidden_states=True)
    
    
    # ============================= direct opt output ======================
    # Access the generated token IDs
    token_ids = opt_outputs.logits.argmax(-1)
    # Decode the token IDs using the tokenizer
    
    print(token_ids[0])
    generated_text = OPT_tokenizer.decode(token_ids[0][2], skip_special_tokens=True)
    # Print the generated text
    print("First Generated Text: ", generated_text)    
    
    # ============================== end of opt output ======================
    
    
    

    # Extract the last hidden state from translator
    translator_last_hidden_state = translator_outputs.decoder_hidden_states[-1]

    # Extract the first hidden state from OPT
    opt_first_hidden_state = opt_outputs.hidden_states[1]
    
    print(translator_last_hidden_state.shape, opt_first_hidden_state.shape)
    data = [(ge.pad(translator_last_hidden_state),ge.pad(opt_first_hidden_state))]
    data_padded, labels_padded, data_masks, labels_masks = ge.pad_and_mask(data,10)
    
    print(data_padded.shape, labels_padded.shape)
    # data_padded, labels_padded, data_masks, labels_masks = ge.pad_and_mask(opt_first_hidden_state,10)
    return data_padded, labels_padded



def create_model(num_layers=2, num_heads=1, dim_feedforward=32, dropout=0.2, lr=0.0010074747982683552, dataset_path: str = "", epochs = 10):
    
    # Create the model, criterion, and optimizer
    model = ge.HiddenStateTransformer(num_layers=num_layers, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model, test_loader = ge.train_model(model,criterion,optimizer,dataset_path,epochs)

    joblib.dump(model, 'transformer_1/orel/general_model.pkl')
    joblib.dump(test_loader, 'transformer_1/orel/general_model_test_loader.joblib')
    
    return model, test_loader


def test(my_model, h_text):
    tras_last_hs_padded, opt_first_hs_padded = generate_test_inputs(h_text)
    my_model.eval()
    with torch.no_grad():
        hidden_states = my_model(tras_last_hs_padded)
        
        criterion = nn.MSELoss()
        validation_loss = criterion(hidden_states, opt_first_hs_padded).item()  # Accumulate validation loss
        
        
        print(f"Validation Loss: {validation_loss:.4f}")
        
        print(hidden_states, hidden_states.shape)

        # Create an instance of the layer you want to modify
        custom_layer = opt_model.base_model.decoder.layers[1]
        # Wrap the layer inside the custom wrapper
        
        # wrapped_layer = CustomLayerWrapper(custom_layer, hidden_states)
        wrapped_layer = CustomLayerWrapper(custom_layer, hidden_states)
        
        # Replace the layer with the wrapped layer
        opt_model.base_model.decoder.layers[1] = wrapped_layer


        # TODO - make this work

        
        # make dummy model

        # Tokenize text to exactly 1024 tokens
        # inputs = OPT_tokenizer("it my", return_tensors="pt",max_length=1024, padding=True)
        
        inputs = OPT_tokenizer(" " * 9, return_tensors="pt")
        
        print(inputs)
        print(inputs["input_ids"].shape)
        # # Create a custom attention mask
        # # Example: Say you want all tokens to be attended to
        # # attention_mask = torch.ones_like(hidden_states[0])
        
        # # attention_mask = torch.ones((1, 1024), dtype=torch.long)
        # attention_mask = torch.ones(inputs['input_ids'].shape, dtype=torch.long)
        
        # print(f"attention_mask.shape = {attention_mask.shape}")

        # # Modify the mask if needed, for example, ignore the last 10 tokens
        # # attention_mask[0, -10:] = 0  # Uncomment and adjust indices as needed

        # # Update the inputs dictionary with the new attention mask
        # inputs['attention_mask'] = attention_mask
        
        
        
        print(inputs["attention_mask"].shape)
        print(inputs)

        outputs = opt_model(**inputs, output_hidden_states=True)

        # # hs = outputs.hidden_states[1]
        # # numpy_array = hs.detach().numpy()
        # # np.save('tensor_data.npy', numpy_array)


        # Access the generated token IDs
        token_ids = outputs.logits.argmax(-1)
        # Decode the token IDs using the tokenizer
        
        print(token_ids[0])
        generated_text = OPT_tokenizer.decode(token_ids[0], skip_special_tokens=True)
        # Print the generated text
        print("Generated Text: ", generated_text)

# ge.test_model(model, test_loader, criterion)

# h_text = "ןגל אב אבא"
h_text = "אני"

# print(tras_last_hs.shape, opt_first_hs.shape)


model: ge.HiddenStateTransformer = joblib.load('transformer_1/orel/model_10Tokens_1/general_model.pkl')
test(model, h_text)


# create_model(1, 1, 128, 0.15, 0.013256841285324495, "resources/up_to_ten_tokens_dataset.pt", 20)

# model: ge.HiddenStateTransformer = joblib.load('transformer_1/orel/general_model.pkl')





# # TODO - FIX -Opt gives the next word[i] basee on token[i]  


# # OPT
# opt_inputs = OPT_tokenizer(h_text, return_tensors="pt")


# # opt_outputs = opt_model(input_ids=opt_inputs.input_ids, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
# opt_outputs = opt_model(**opt_inputs, output_hidden_states=True)


# # ============================= direct opt output ======================
# # Access the generated token IDs
# token_ids = opt_outputs.logits.argmax(-1)
# # Decode the token IDs using the tokenizer
# print(token_ids[0])
# generated_text = OPT_tokenizer.decode(token_ids[0], skip_special_tokens=True)
# # Print the generated text
# print("Last Generated Text: ", generated_text)

# # ============================== end of opt output ======================