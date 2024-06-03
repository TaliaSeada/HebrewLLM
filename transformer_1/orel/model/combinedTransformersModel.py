# import torch
# import torch.nn as nn
# import joblib
# from transformers import AutoTokenizer, AutoModel, MarianTokenizer,MarianMTModel, AutoTokenizer, OPTForCausalLM, AutoModelForSeq2SeqLM
# from generalTransformer import generate_test_inputs
# import TransGeneralEmbeddingToOPT as ge
# import pandas as pd



# device = "cuda" if torch.cuda.is_available() else "cpu"

# # # Hebrew to english translator
# # translator_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"

# # translator_tokenizer = MarianTokenizer.from_pretrained(translator_model_name)
# # translator_model = MarianMTModel.from_pretrained(translator_model_name)

# # # OPT model
# # OPT_model_name = "facebook/opt-350m"
# # OPT_tokenizer = AutoTokenizer.from_pretrained(OPT_model_name)
# # opt_model = OPTForCausalLM.from_pretrained(OPT_model_name).to(device)

# # # English to Hebrew translator
# # model_name = "Helsinki-NLP/opus-mt-en-he"
# # translator_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# # translator_tokenizer = AutoTokenizer.from_pretrained(model_name)


# class CombinedModel(nn.Module):
#     def __init__(self, transformer1, transformer2, llm):
#         super(CombinedModel, self).__init__()
#         self.transformer1 = transformer1
#         self.llm = llm
#         self.transformer2 = transformer2
        
#         # Freeze LLM parameters
#         for param in self.llm.parameters():
#             param.requires_grad = False
    
#     def forward(self, x):
#         x = self.transformer1(x)
#         # Ensure LLM does not compute gradients
#         with torch.no_grad():
#             x = self.llm(x).last_hidden_state
#         x = self.transformer2(x)
#         return x

#     def test():
#         pass
    
#     def hebrew_to_input(self,h_text, hebrew_translator_tokenizer,hebrew_translator_model):
#             # Translator
#         inputs = hebrew_translator_tokenizer(h_text, return_tensors="pt")
        
#         # Encode the source text
#         generated_ids = hebrew_translator_model.generate(inputs.input_ids)

#         # Append hidden states
#         translator_outputs = hebrew_translator_model(input_ids=inputs.input_ids, decoder_input_ids=generated_ids, output_hidden_states=True)
        
#         # Extract the last hidden state from translator
#         translator_last_hidden_state = translator_outputs.decoder_hidden_states[-1]

#         data = [(ge.pad(translator_last_hidden_state),)]
#         data_padded, labels_padded, data_masks, labels_masks = ge.pad_and_mask(data, False)

#         return data_padded


# def train_combined_model(model, criterion, optimizer, epochs, batch_size, dataset_path):
#     # Hebrew to english translator
#     He_En_translator_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
#     He_En_tokenizer = MarianTokenizer.from_pretrained(He_En_translator_name)
#     He_En_translator_model = MarianMTModel.from_pretrained(He_En_translator_name)
    
#     # English to Hebrew translator
#     En_He_translator_name = "Helsinki-NLP/opus-mt-en-he"
#     En_He_translator_model = AutoModelForSeq2SeqLM.from_pretrained(En_He_translator_name)
#     En_He_tokenizer = AutoTokenizer.from_pretrained(En_He_translator_name)
    
        
#     df = pd.read_csv('wikipedia_data.csv')

#     # print(type(df['Hebrew sentence']))


#     for index, row in df.iterrows():
#         data = row[0]
#         label = row[0] + " " + row[1]
    
#     # # Adjust DataLoader batch size
#     # train_loader, val_loader, test_loader = create_data_loaders(dataset_path, batch_size)
    
#     # print("Data Loaders created!")
    
#     # # Train the model
#     # for epoch in range(epochs):
#     #     model.train()  # Set the model to training mode
#     #     train_loss = 0
#     #     for data, labels, data_masks, labels_masks in train_loader:

#     #         optimizer.zero_grad()  # Zero out any gradients from previous steps

#     #         output = model(data, src_key_padding_mask=data_masks)  # Ensure masks are used
#     #         loss = criterion(output, labels)  # Calculate loss
#     #         loss.backward(retain_graph=True)  # Backpropagate the error
#     #         optimizer.step()  # Update model parameters
#     #         train_loss += loss.item()
#     #     train_loss /= len(train_loader)  # Average the loss over the batch

#     #     # Validation phase
#     #     model.eval()  # Set the model to evaluation mode
#     #     validation_loss = 0
#     #     for data, labels, data_masks, labels_masks in val_loader:

#     #         with torch.no_grad():  # No gradient calculation
#     #             output = model(data, src_key_padding_mask=data_masks)  # Use masks during validation as well
#     #             validation_loss += criterion(output, labels).item()  # Accumulate validation loss
#     #     validation_loss /= len(val_loader)
        
#     #     print(f"Epoch {epoch+1}, Validation Loss: {validation_loss:.4f}")
    
#     # return model, test_loader, train_loader, val_loader
            
            

# # Transformer 1
# t1 = joblib.load('transformer_1/orel/models/model_name.pkl')


# # LLM model
# llm_model_name = "facebook/opt-350m"
# llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
# llm = OPTForCausalLM.from_pretrained(llm_model_name).to(device)

# # Transformer 2
# t2 = joblib.load('transformer_2/model_name.pkl')




# combined_model = CombinedModel(transformer1=t1,transformer2=t2,llm=llm)

