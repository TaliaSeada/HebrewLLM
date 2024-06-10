from data.dataManipulation import pad, pad_and_mask
from model.HiddenStateTransformer import HiddenStateTransformer, train_model, test_model
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


class CustomLayerWrapper2(nn.Module):
    def __init__(self, layer, hidden_states):
        super().__init__()
        self.layer = layer
        self.hs = hidden_states  # transformer's result

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # Modify hidden_states before passing them to the layer
        modified_hidden_states = your_input_modification(self.hs)

        # Ensure that attention_mask and any other necessary arguments are forwarded
        return self.layer(modified_hidden_states, attention_mask, **kwargs)

    
    
def generate_test_inputs(curr_song):
    # Translator
    inputs = translator_tokenizer(curr_song, return_tensors="pt")
    
    # Encode the source text
    generated_ids = translator_model.generate(inputs.input_ids)

    # Translation to english
    english_text = translator_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print(f"Tokenizer1 : {generated_ids[0]}")
    
    english_text = english_text if english_text[-1] != '.' else english_text[:-1]
    print(english_text)
    
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
    
    print(f"OPT tokens: {token_ids[0]}")
    generated_text = OPT_tokenizer.decode(token_ids[0], skip_special_tokens=True)
    # Print the generated text
    print("OPT Generated Text: ", generated_text)    
    
    # ============================== end of opt output ======================
    
    
    

    # Extract the last hidden state from translator
    translator_last_hidden_state = translator_outputs.decoder_hidden_states[-1]

    # Extract the first hidden state from OPT
    opt_first_hidden_state = opt_outputs.hidden_states[1]
    
    print(translator_last_hidden_state.shape, opt_first_hidden_state.shape)
    data = [(pad(translator_last_hidden_state),pad(opt_first_hidden_state))]
    data_padded, labels_padded, data_masks, labels_masks = pad_and_mask(data,10)
    
    print(data_padded.shape, labels_padded.shape)
    # data_padded, labels_padded, data_masks, labels_masks = ge.pad_and_mask(opt_first_hidden_state,10)
    return data_padded, labels_padded, len(generated_ids[0])



def create_model(criterion, model_path: str, loader_path: str, model_num, num_layers=2, num_heads=1, dim_feedforward=32, dropout=0.2, lr=0.0010074747982683552, dataset_path: str = "", epochs = 10, batch_size = 16):
    
    print(f"model_path = {model_path}, num_layers = {num_layers}, num_heads = {num_heads}, dim_feedforward = {dim_feedforward}, dropout = {dropout}, lr = {lr}, batch_size = {batch_size}")
    # Create the model, criterion, and optimizer
    model = HiddenStateTransformer(num_layers=num_layers, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model, test_loader, train_loader, val_loader = train_model(model,criterion,optimizer,dataset_path,epochs,batch_size)

    joblib.dump(model, model_path)
    joblib.dump(test_loader, f"{loader_path}/testLoaders/model_{model_num}.joblib")
    joblib.dump(train_loader, f"{loader_path}/trainLoaders/model_{model_num}.joblib")
    joblib.dump(val_loader, f"{loader_path}/valLoaders/model_{model_num}.joblib")


    return model, test_loader


def get_top_results(llm, tokenizer, first_hs, topN=3):
        # Create an instance of the layer you want to modify
        custom_layer = llm.base_model.decoder.layers[1]
        # Wrap the layer inside the custom wrapper
        
        # wrapped_layer = CustomLayerWrapper(custom_layer, hidden_states)
        wrapped_layer = CustomLayerWrapper(custom_layer, first_hs)
        
        # Replace the layer with the wrapped layer
        llm.base_model.decoder.layers[1] = wrapped_layer

        # make dummy model
        inputs = tokenizer(" " * 14, return_tensors="pt")
        
        outputs = llm(**inputs)

        
        # Get the top 3 logits for each position
        top_n_logits = torch.topk(outputs.logits, topN, dim=-1).indices
        
        # Decode the top 3 logits into words
        top_n_words = []
        for i in range(top_n_logits.size(1)):
            words = [tokenizer.decode([token_id.item()]) for token_id in top_n_logits[0, i]]
            top_n_words.append(words)
        
        return top_n_words
    
    
def test(my_model, h_text):
    tras_last_hs_padded, opt_first_hs_padded, input_tokens_len = generate_test_inputs(h_text)
    my_model.eval()
    with torch.no_grad():
        hidden_states = my_model(tras_last_hs_padded)
        
        criterion = nn.MSELoss()
        validation_loss = criterion(hidden_states, opt_first_hs_padded).item()  # Accumulate validation loss
        
        
        print(f"Validation Loss: {validation_loss:.4f}")
        
        top3_direct = get_top_results(opt_model, OPT_tokenizer, opt_first_hs_padded)

        OPT_tokenizer2 = AutoTokenizer.from_pretrained(OPT_model_name)
        opt_model2 = OPTForCausalLM.from_pretrained(OPT_model_name).to(device)

        top3_my_model = get_top_results(opt_model2, OPT_tokenizer2, hidden_states)
        
        for i in range(len(top3_direct)):
            flag = ""
            if i == input_tokens_len - 3:
                flag += "* "
            begin = f"Token position {i}: actutal({top3_direct[i]})"
            spaces = " " * (60 - len(begin) - len(flag)) + "|" + " " * 5
            end = f"my_model({top3_my_model[i]})"
            print(flag + begin + spaces + flag + end)


num = 20
model_path = f'transformer_1/orel/pretrainedModels/models/15Tokens/model_wiki_{num}.pkl'
loader_path = f'transformer_1/orel/pretrainedModels/loaders/15Tokens/'


# create_model(model_path, loader_path,num,1,4,256,0.25,0.002676001187706025,'resources/datasets/dataset_wiki_up_to_15_tokens.pt',10,32)

# create_model(model_path, loader_path,num,1,8,256,0.15,0.004827586123698931,'resources/datasets/dataset_wiki_up_to_15_tokens.pt',15,32)

# ge.test_model(model, test_loader, criterion)

# # h_text = "ןגל אב אבא"
# h_text = "אני רוצה לישון הרבה מאוד"

# # print(tras_last_hs.shape, opt_first_hs.shape)

# # print("\n\n ========== 15 Tokens Model ==========\n\n")
# # model: ge.HiddenStateTransformer = joblib.load(model_path)
# # test(model, h_text)


# # # create_model(1, 1, 128, 0.15, 0.013256841285324495, "resources/up_to_ten_tokens_dataset.pt", 20)
# print("\n\n ========== 10 Tokens Model ==========\n\n")
# model2 = joblib.load('transformer_1/orel/model_10Tokens_1/general_model0.pkl')
# test(model2, h_text)

# criterion = nn.MSELoss()
# ge.test_model2(model, test_loader,criterion)





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



num = 20
model_path = f'transformer_1/orel/pretrainedModels/models/15Tokens/model_wiki_{num}.pkl'
loader_path = f'transformer_1/orel/pretrainedModels/loaders/15Tokens/'
