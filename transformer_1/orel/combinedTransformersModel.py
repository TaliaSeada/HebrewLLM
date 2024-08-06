# import torch
# import torch.nn as nn
# import torch.optim as optim
# import joblib
# from transformers import AutoTokenizer, AutoModel, MarianTokenizer, MarianMTModel, AutoTokenizer, OPTForCausalLM
# import pandas as pd
# from torch.utils.data import DataLoader, Dataset
# from data.dataManipulation import pad, pad_and_mask
# from model.HiddenStateTransformer import HiddenStateTransformer, HiddenStateTransformer2, train_model, test_model
# from generalTransformer import CustomLayerWrapper, CustomLayerWrapper2
# import torch.nn.functional as F
# import math

# import os

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# def generate_with_logits(model, tokenizer, ids):
#     # Ensure the attention mask is correctly shaped
#     attention_mask = torch.ones((1, 15))
#     ids['attention_mask'] = attention_mask
    
#     # Prepare decoder_input_ids, starting with the <pad> token
#     decoder_input_ids = torch.full(
#         (ids.input_ids.size(0), 15), tokenizer.pad_token_id, dtype=torch.long
#     )

#     # Concatenate with input_ids shifted right
#     decoder_input_ids = torch.cat([decoder_input_ids, ids.input_ids], dim=1)

#     # print(f"decoder_input_ids = {decoder_input_ids},\nShape = {decoder_input_ids.shape}")

#     # Forward pass to get the logits
#     outputs = model(
#         input_ids=ids.input_ids,
#         attention_mask=attention_mask,
#         decoder_input_ids=decoder_input_ids,
#         output_hidden_states=True
#     )

#     logits = outputs.logits
    
#     # Access the generated token IDs
#     token_ids = outputs.logits.argmax(-1)
#     # # Decode the token IDs using the tokenizer
#     generated_text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
    
#     print(f"Output token_ids {token_ids} = {generated_text}")

#     logits = logits.requires_grad_()

#     # Apply softmax to get probabilities
#     probabilities = F.log_softmax(logits, dim=-1)

#     return logits, probabilities


# class CombinedModel(nn.Module):
#     def __init__(self, transformer1, transformer2, llm):
#         super(CombinedModel, self).__init__()
#         self.transformer1 = transformer1
#         self.llm: OPTForCausalLM = llm
#         self.transformer2 = transformer2

#         # Freeze LLM parameters
#         for param in self.llm.parameters():
#             param.requires_grad = False

#     def forward(self, text, tokenizer1, translator1, llm_tokenizer, tokenizer2, translator2):

#         with torch.no_grad():
#             # Get the final embedding of translator1 for the text input (language1)
#             x, _ = hebrew_to_input(text, tokenizer1, translator1)
#             # print(f"Text = {text}\nInput = {x}")

#         # Transform it to the llm initial embeddings for the sentence in language2 
#         x = self.transformer1(x)
        
#         # print(f"Transformer 1 output = {x}")

#         # Ensure LLM does not compute gradients
#         with torch.no_grad():
#             # '''==================== Check this ================================='''
#             # llm_model_name = "facebook/opt-350m"
#             # llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
#             # self.llm = OPTForCausalLM.from_pretrained(llm_model_name)
#             # '''=================================================================='''

#             # Trick llm by giving it a dummy that contain the desired number of tokens (In our case 15)
#             # and replace the first layer as it got other word embedding.
#             inputs = llm_tokenizer(" " * 14, return_tensors="pt")

#             # Inject our initial embeddings to the first layer of the llm
#             self.llm = self.inject_layer(model=self.llm, layer_hs=x, layer_num=1, name="decoder")
                        
#             # Calculates the final embeddings
#             outputs = self.llm(**inputs, output_hidden_states=True)

#             # Lastly, get the final embeddings of the llm for our injected input
#             llm_last_hidden_state = outputs.hidden_states[-1]
            
#             # print(f"LLM output = {llm_last_hidden_state}")
            

#         # Transform it (embeddings of output sentence in language 2) to the initial embeddings of translator2
#         x = self.transformer2(llm_last_hidden_state)

#         # ================================= Problem is here ================================
        
#         # print(f"Transformer 2 output = {x}")

#         with torch.no_grad():
#             # Do the same trick as before
#             inputs = tokenizer2("a " * 14, return_tensors="pt")

#             # print(f"inputs.input_ids = {inputs.input_ids}")

#             # # English to Hebrew translator
#             En_He_model_name = "Helsinki-NLP/opus-mt-en-he"
#             # # En_He_tokenizer = MarianTokenizer.from_pretrained(En_He_model_name)
#             translator2 = MarianMTModel.from_pretrained(En_He_model_name)

#             # original = translator2.base_model.encoder.layers[1]

#             self.inject_layer(model=translator2, layer_hs=x, layer_num=1, name="encoder")

#             # print(f"New layer hs = {translator2.model.encoder.layers[1].hs}")

#             # return the probability for each word in the dictionary for each vector in the final embeddings of translator2
#             l, p = generate_with_logits(translator2, En_He_tokenizer, inputs)

#             # TODO - why  the logits stay the same
#         # =================================== Problem ==================================

#             # print(attention_mask,l.shape,p.shape)
#             # print(p)
#         return l, p

#     def test(self):
#         pass

#     def inject_layer(self, model, layer_hs, layer_num, name="decoder"):
#         if name == "decoder":
#             original_layer = model.base_model.decoder.layers[layer_num]
#             wrapped_layer = CustomLayerWrapper(original_layer, layer_hs)
#         else:
#             original_layer = model.model.encoder.layers[layer_num]
#             wrapped_layer = CustomLayerWrapper2(original_layer, layer_hs)

#         # Wrap the layer inside the custom wrapper

#         # Replace the layer with the wrapped layer
#         if name == "decoder":
#             model.base_model.decoder.layers[layer_num].hs = layer_hs
#             model.base_model.decoder.layers[layer_num] = wrapped_layer
#         else:
#             model.model.encoder.layers[layer_num].hs = layer_hs
#             model.model.encoder.layers[layer_num] = wrapped_layer

#         return model


# def hebrew_to_input(h_text, hebrew_translator_tokenizer, hebrew_translator_model):
#     # Translator
#     inputs = hebrew_translator_tokenizer(h_text, return_tensors="pt")

#     # Encode the source text
#     generated_ids = hebrew_translator_model.generate(inputs.input_ids)

#     # print(f"Hebrew input ids = {len(generated_ids[0])}")

#     clean_token_num = len(generated_ids[0]) - 2

#     # for t in generated_ids:
#     #     print(f"translated token1 {t} = {hebrew_translator_tokenizer.convert_ids_to_tokens(t)}")

#     # Append hidden states
#     translator_outputs = hebrew_translator_model(input_ids=inputs.input_ids, decoder_input_ids=generated_ids,
#                                                  output_hidden_states=True)

#     # Extract the last hidden state from translator
#     translator_last_hidden_state = translator_outputs.decoder_hidden_states[-1]

#     data = [(pad(translator_last_hidden_state),)]
#     data_padded, labels_padded, data_masks, labels_masks = pad_and_mask(data, False)

#     # print(f"data_padded = {data_padded}")
#     return data_padded, clean_token_num


# def translate_to_english(model, tokenizer, sentence):
#     inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
#     translated_tokens = model.generate(**inputs)
#     translated_sentence = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
#     return translated_sentence


# def translate_to_hebrew(model, tokenizer, sentence):
#     inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
#     translated_tokens = model.generate(**inputs)
#     return translated_tokens


# def train_combined_model(dataset_path, stop_index, model: CombinedModel, He_En_model, En_He_model, He_En_tokenizer,
#                          En_He_tokenizer, criterion, optimizer, epochs):
#     df = pd.read_csv(dataset_path)

#     # Train the model
#     for epoch in range(epochs):
#         model.train()  # Set the model to training mode
#         train_loss = 0

#         for index, row in df.iterrows():
#             if index > stop_index:
#                 break
#             hebrew_sentence = row['Hebrew sentence']
#             target_hebrew_sentence = row['Hebrew sentence'] + " " + row['label']

#             # TODO ==== Black box -> New Hebrew Sentence outputs ====

#             # Outputs the first layer of the En_He_translator using transformer1
#             logits, distribution = model(
#                 hebrew_sentence,
#                 He_En_tokenizer,
#                 He_En_model,
#                 llm_tokenizer,
#                 En_He_tokenizer,
#                 En_He_model)

#             # print(f"hebrew_ids = {generated_output[0]}")

#             # # Translation for us
#             # hebrew_sentence = En_He_tokenizer.decode(generated_output[0], skip_special_tokens=True)
#             # print(f"generated_text = {hebrew_sentence}")

#             # ==== calc loss =====

#             # Get the tokens for the target sentence
#             target_ids = En_He_tokenizer(text_target=target_hebrew_sentence, return_tensors="pt")

#             # print(target_ids[:,:15].squeeze(0).shape)

#             # TODO - Pad the tensors to 15 so their size will match

#             # Calculate the loss
#             # print(f"shape(logits) = {logits.squeeze(0)[1:2,:]}")
#             # print(f"logits = {logits.squeeze(0)[1:2,:]}")

#             # Get the top 3 logits for each position
#             top_n_logits = torch.topk(logits, 3, dim=-1).indices

#             # Decode the top 3 logits into words
#             top_n_words = []
#             for i in range(top_n_logits.size(1)):
#                 words = [En_He_tokenizer.decode([token_id.item()]) for token_id in top_n_logits[0, i]]
#                 top_n_words.append(words)

#             # print(f"sentence {index} = {top_n_words}")

#             # This is the vector that represent the word in index 1
#             for t in logits.squeeze(0)[1:2]:
#                 # print(f"translated logits {t} = {En_He_tokenizer.convert_ids_to_tokens(t)}")
#                 print(f"logits {t}")

#             # print(logits.squeeze(0)[1:,:].shape, target_ids.squeeze(0)[:].shape)

#             max_left = logits.squeeze(0)[1:, :].shape[0]
#             max_right = target_ids.input_ids.squeeze(0).shape[0]

#             desired_len = min(max_left, max_right, 14)

#             loss = criterion(logits.squeeze(0)[1:desired_len + 1, :], target_ids.input_ids.squeeze(0)[:desired_len])

#             # Back Propagation
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()
#             # print(f"Loss: {loss}")

#         train_loss /= stop_index
#         print(f"=========================== Epoch {epoch}, Loss = {train_loss} ===========================")


# def my_cross_entropy(dist, target):
#     # return sum([(math.log(dist[0,index,token_id])) if index < 15 else 0 for index, token_id in enumerate(target.squeeze(0))])
#     # print(target.view(-1))
#     return F.nll_loss(dist, target)


# def save_model(model, to_path):
#     joblib.dump(model, to_path)


# # Hebrew to english translator
# He_En_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
# He_En_tokenizer = MarianTokenizer.from_pretrained(He_En_model_name)
# He_En_translator_model = MarianMTModel.from_pretrained(He_En_model_name)

# # Transformer 1
# t1 = joblib.load('transformer_1/orel/pretrainedModels/models/10Tokens/general_model.pkl')
# # t1 = joblib.load('C:\\Users\\talia\\PycharmProjects\\HebrewLLM\\transformer_1\\orel\\pretrainedModels\\models\\10Tokens'
# #                  '\\general_model.pkl')

# # LLM model
# llm_model_name = "facebook/opt-350m"
# llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
# llm = OPTForCausalLM.from_pretrained(llm_model_name)

# # Transformer 2
# # t2 = joblib.load('transformer_2/model_name.pkl')
# # t2 = HiddenStateTransformer2(input_size=512,output_size=512, num_layers=1, num_heads=2, dim_feedforward=256, dropout=0.15)
# t2 = joblib.load('C:\\Users\\orelz\\OneDrive\\שולחן העבודה\\work\\Ariel\\HebrewLLM\\transformer_2\\pretranedModels\\models\\15Tokens\\model_15_tokens_talia.pkl')
# # t2 = joblib.load(
# #     'C:\\Users\\talia\\PycharmProjects\\HebrewLLM\\transformer_2\\pretranedModels\\models\\15Tokens'
# #     '\\model_15_tokens_talia.pkl')

# # English to Hebrew translator
# En_He_model_name = "Helsinki-NLP/opus-mt-en-he"
# En_He_tokenizer = MarianTokenizer.from_pretrained(En_He_model_name)
# En_He_translator_model = MarianMTModel.from_pretrained(En_He_model_name)

# combined_model = CombinedModel(transformer1=t1, transformer2=t2, llm=llm)

# # for name, param in combined_model.named_parameters():
# #     if param.requires_grad:
# #         print(f'{name} requires grad')


# # optimizer = optim.Adam(combined_model.parameters(), lr=0.001)
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, combined_model.parameters()), lr=0.01)

# criterion = nn.CrossEntropyLoss()
# # criterion = my_cross_entropy

# path = 'transformer_1/orel/wikipedia_data_15.csv'
# # path = 'C:\\Users\\talia\\PycharmProjects\\HebrewLLM\\wikipedia_data.csv'

# train_combined_model(path,
#                      40,
#                      combined_model,
#                      He_En_translator_model,
#                      En_He_translator_model,
#                      He_En_tokenizer,
#                      En_He_tokenizer,
#                      criterion,
#                      optimizer,
#                      10)

# # num = 1
# # save_model(combined_model, f'transformer_1/orel/pretrainedModels/models/combined/model_wiki_{num}.pkl')




# ============================================== try 2 ========================================
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from transformers import AutoTokenizer, AutoModel, MarianTokenizer, MarianMTModel, AutoTokenizer, OPTForCausalLM
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from data.dataManipulation import pad, pad_and_mask
from model.HiddenStateTransformer import HiddenStateTransformer, HiddenStateTransformer2, train_model, test_model
from generalTransformer import CustomLayerWrapper, CustomLayerWrapper2
import torch.nn.functional as F
import math

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class CombinedModel(nn.Module):
    def __init__(self, tokenizer1, translator1, transformer1, llm_tokenizer, llm, transformer2, tokenizer2, translator2):
        super(CombinedModel, self).__init__()
        self.tokenizer1 = tokenizer1
        self.translator1 = translator1
        
        self.transformer1 = transformer1
        
        self.llm_tokenizer = llm_tokenizer
        self.llm: OPTForCausalLM = llm
        
        self.transformer2 = transformer2
        
        self.tokenizer2 = tokenizer2
        self.translator2: MarianMTModel = translator2

        # Freeze Translator1 parameters
        for param in self.translator1.parameters():
            param.requires_grad = False
            
        # Freeze LLM parameters
        for param in self.llm.parameters():
            param.requires_grad = False
    
        # Freeze Translator2 parameters
        for param in self.translator2.parameters():
            param.requires_grad = False
            

        original_layer = self.llm.base_model.decoder.layers[1]
        wrapped_layer = CustomLayerWrapper(original_layer, None)
        self.llm.base_model.decoder.layers[1] = wrapped_layer
        
        original_layer2 = self.translator2.model.encoder.layers[1]
        wrapped_layer2 = CustomLayerWrapper2(original_layer2, None)
        self.translator2.model.encoder.layers[1] = wrapped_layer2


    def forward(self, text, target=None):
        
        # Get the final embedding of translator1 for the text input (language1)
        x, _ = self.hebrew_to_input(text)
        # print(f"Text = {text}\nInput = {x}")
        
        if x.shape[1] > 15:
            return None
        
        # Transform it to the llm initial embeddings for the sentence in language2 
        x = self.transformer1(x)
        
        # # Ensure LLM does not compute gradients

        # Trick llm by giving it a dummy that contain the desired number of tokens (In our case 15)
        # and replace the first layer as it got other word embedding.
        inputs = self.llm_tokenizer(" " * 14, return_tensors="pt")

        # Inject our initial embeddings to the first layer of the llm
        self.inject_layer(layer_hs=x, layer_num=1, name="decoder")
                    
        # Calculates the final embeddings
        outputs = self.llm(**inputs, output_hidden_states=True)

        # # Getting the predicted tokens (usually the last hidden state contains the logits for token predictions)
        # predicted_tokens = outputs.logits.argmax(dim=-1)

        # # Convert tokens to words
        # predicted_words = self.llm_tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)

        # print(f"predicted_words = {predicted_words}")

        # Lastly, get the final embeddings of the llm for our injected input
        llm_last_hidden_state = outputs.hidden_states[-1]


        # Transform it (embeddings of output sentence in language 2) to the initial embeddings of translator2
        x = self.transformer2(llm_last_hidden_state)

        # Inject our initial embeddings to the first layer of Transformer2
        self.inject_layer(layer_hs=x, layer_num=1, name="encoder")
        
        # return the probability for each word in the dictionary for each vector in the final embeddings of translator2
        if target is None:
            q = self.generate_predicted_distribution(text)
        else:
            q = self.generate_predicted_distribution(target)
            
        return q


    def generate(self, hebrew_text: str):
        logits = self(hebrew_text)
        token_ids = logits.argmax(-1)

        generated_text = self.tokenizer2.decode(token_ids[0], skip_special_tokens=True)

        return generated_text


    def inject_layer(self, layer_hs, layer_num, name="decoder"):

        # Replace the layer with the wrapped layer
        if name == "decoder":
            self.llm.base_model.decoder.layers[layer_num].hs = layer_hs
        else:
            self.translator2.model.encoder.layers[layer_num].hs = layer_hs


    def hebrew_to_input(self,h_text):
        # Translator
        inputs = self.tokenizer1(h_text, return_tensors="pt")

        # Encode the source text
        generated_ids = self.translator1.generate(inputs.input_ids)

        # print(f"Hebrew input ids = {len(generated_ids[0])}")

        clean_token_num = len(generated_ids[0]) - 2

        # for t in generated_ids:
        #     print(f"translated token1 {t} = {hebrew_translator_tokenizer.convert_ids_to_tokens(t)}")

        # Append hidden states
        translator_outputs = self.translator1(input_ids=inputs.input_ids, decoder_input_ids=generated_ids,
                                                    output_hidden_states=True)

        # Extract the last hidden state from translator
        translator_last_hidden_state = translator_outputs.decoder_hidden_states[-1]

        data = [(pad(translator_last_hidden_state),)]
        data_padded, labels_padded, data_masks, labels_masks = pad_and_mask(data, False)

        # print(f"data_padded = {data_padded}")
        return data_padded, clean_token_num


    def generate_predicted_distribution(self, text):
        
        # Get the tokens for the target sentence
        known_target_ids = self.tokenizer2(text_target=text, return_tensors="pt").input_ids
        
        # Get the padding token ID
        pad_token_id = self.tokenizer2.pad_token_id

        # Create a tensor filled with pad token IDs, with the desired length
        max_length = 15
        # batch_size = known_target_ids.size(0)
        # decoder_input_ids = torch.full((batch_size, max_length - known_target_ids.shape[1]), pad_token_id, dtype=torch.long)

            
        # # Trick Translator by giving it a dummy that contain the desired number of tokens (In our case 15)
        # # and replace the first layer as it got other word embedding.
        inputs = self.tokenizer2("a " * 14, return_tensors="pt")
        
        
        # Ensure the attention mask is correctly shaped
        attention_mask = torch.ones((1, 15))
        inputs['attention_mask'] = attention_mask
        
        decoder_len = max((max_length - known_target_ids.shape[1]), 0)
        
        # Prepare decoder_input_ids, starting with the <pad> token
        decoder_input_ids = torch.full(
            (inputs.input_ids.size(0),  decoder_len), self.tokenizer2.pad_token_id, dtype=torch.long
        )

        # Concatenate with input_ids shifted right
        decoder_input_ids = torch.cat([known_target_ids, decoder_input_ids], dim=1)

        # print(f"decoder_input_ids = {decoder_input_ids},\nShape = {decoder_input_ids.shape}")

        # Forward pass to get the logits
        outputs = self.translator2(
            input_ids=inputs.input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True
        )
        
        # print(f"outputs.keys() = {outputs.keys()}")
        
        # Getting the logits (usually the last hidden state contains the logits for token predictions)
        logits = outputs.logits
        
        # # Apply softmax to get probabilities
        # q = torch.softmax(logits, dim=-1)
        
        q = logits
        # print(f"q = {q}")
        
        # =========================================================
        
        # model_output = self.translator2.generate(
        #     inputs.input_ids,
        #     return_dict_in_generate=True,
        #     output_scores=True
        # )
        
        # # Predicted distribution
        # q = torch.stack(model_output.scores, dim=1)[0].softmax(dim=-1)
        
        # print(f"q = {q}")
        # =========================================================

        return q

    # def generate_with_logits(self):
    #     # Trick Translator by giving it a dummy that contain the desired number of tokens (In our case 15)
    #     # and replace the first layer as it got other word embedding.
    #     inputs = self.tokenizer2("a " * 14, return_tensors="pt")
            
    #     # Ensure the attention mask is correctly shaped
    #     attention_mask = torch.ones((1, 15))
    #     inputs['attention_mask'] = attention_mask
        
    #     # Prepare decoder_input_ids, starting with the <pad> token
    #     decoder_input_ids = torch.full(
    #         (inputs.input_ids.size(0), 15), self.tokenizer2.pad_token_id, dtype=torch.long
    #     )

    #     # Concatenate with input_ids shifted right
    #     decoder_input_ids = torch.cat([decoder_input_ids, inputs.input_ids], dim=1)

    #     # print(f"decoder_input_ids = {decoder_input_ids},\nShape = {decoder_input_ids.shape}")

    #     # Forward pass to get the logits
    #     outputs = self.translator2(
    #         input_ids=inputs.input_ids,
    #         attention_mask=attention_mask,
    #         decoder_input_ids=decoder_input_ids,
    #         output_hidden_states=True
    #     )

    #     logits = outputs.logits
        
    #     # Access the generated token IDs
    #     token_ids = outputs.logits.argmax(-1)
    #     # # Decode the token IDs using the tokenizer
    #     generated_text = self.tokenizer2.decode(token_ids[0], skip_special_tokens=True)
        
    #     print(f"Output token_ids {token_ids} = {generated_text}")

    #     logits = logits.requires_grad_()

    #     # Apply softmax to get probabilities
    #     probabilities = F.log_softmax(logits, dim=-1)

    #     return logits, probabilities


def translate_to_english(model, tokenizer, sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model.generate(**inputs)
    translated_sentence = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_sentence


def translate_to_hebrew(model, tokenizer, sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model.generate(**inputs)
    return translated_tokens


def train_combined_model(dataset_path, stop_index, model: CombinedModel, He_En_model, En_He_model, He_En_tokenizer,
                         En_He_tokenizer, criterion, optimizer, epochs):
    df = pd.read_csv(dataset_path)
    print_every = 350
    
    # Train the model
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        train_loss = 0
        counter = 0

        # before_params = model.transformer2.parameters()
        
        for index, row in df.iterrows():
            if index > stop_index:
                break
            hebrew_sentence = row['Hebrew sentence']
            target_hebrew_sentence = row['Hebrew sentence'] + " " + row['label']
            
            # if index % print_every == 0:
            #     curr_loss = train_loss / counter if counter > 0 else train_loss
            #     print(f"index = {index}, current loss = {curr_loss}")

            # =================== Calculate the loss ===================
            
            # Outputs predicted distribution for each token
            q = model(hebrew_sentence, target_hebrew_sentence)
            
            if q is None:
                continue
            
            counter += 1

            # ==== calc loss =====

            # Get the tokens for the target sentence
            target_ids = En_He_tokenizer(text_target=target_hebrew_sentence, return_tensors="pt")


            # print(f"target_ids = {target_ids}")
            # print(f"q.shape = {q.shape}")
            max_left = q[0, 1:, :].shape[0]
            max_right = target_ids.input_ids.squeeze(0).shape[0]

            # desired_len = min(max_left, max_right, 14)
            desired_len = min(max_left, max_right, 2)            # <<<<<<<<<<< Change!! it learns only 2 words
            
            actual = q[0,1:desired_len + 1, :]
            expected = target_ids.input_ids.squeeze(0)[:desired_len]
            
            actual.requires_grad_()
            # expected.requires_grad_()
            
            # # One-hot encode the expected tokens
            # num_classes = actual.shape[-1]
            # expected_one_hot = one_hot_encode(expected, num_classes)
            
            # print(f"actual_log_probs = {actual_log_probs}")
            # print(f"expected_one_hot = {expected_one_hot}")

            # print(f"expected_one_hot.shape = {expected_one_hot.shape}")
            # print(f"sum = {expected_one_hot.sum()}")
            
            # Calculate the loss
            # loss = criterion(actual, expected_one_hot)
            
            # if index % print_every == 0:
                # # Get the top 3 logits for each position
                # top_n_logits = torch.topk(q, 3, dim=-1).indices

                # # Decode the top 3 logits into words
                # top_n_words = []
                # for i in range(top_n_logits.size(1)):
                #     words = [En_He_tokenizer.decode([token_id.item()]) for token_id in top_n_logits[0, i]]
                #     top_n_words.append(words)
                # print(f"sentence {index} = {top_n_words}")
            
            loss = criterion(actual, expected)

            # Back Propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # print(f"Loss: {loss}")
            
            # after_params = model.transformer2.parameters()

            # print(f"params are equals? {before_params == after_params}")

            
        train_loss /= counter
        print(f"=========================== Epoch {epoch}, Loss = {train_loss} ===========================")
    # save_model(model, "combined_model_1.pkl")
    return train_loss

def save_model(model, to_path):
    joblib.dump(model, to_path)

def one_hot_encode(indices, num_classes):
    return nn.functional.one_hot(indices, num_classes=num_classes).float()




lr=0.006334926670051613
train_size = 750


# # print(f"======================= transformer1: old, transformer2: old, lr: {lr}, train_size: {train_size} =======================")
# # print(f"======================= transformer1: new, transformer2: old, lr: {lr}, train_size: {train_size} =======================")
# # print(f"======================= transformer1: old, transformer2: none, lr: {lr}, train_size: {train_size} =======================")
# # print(f"======================= transformer1: new, transformer2: none, lr: {lr}, train_size: {train_size} =======================")
# print(f"======================= transformer1: none, transformer2: none, lr: {lr}, train_size: {train_size} =======================")



# Hebrew to english translator
He_En_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
He_En_tokenizer = MarianTokenizer.from_pretrained(He_En_model_name)
He_En_translator_model = MarianMTModel.from_pretrained(He_En_model_name)

# Transformer 1
# t1 = HiddenStateTransformer(input_size=1024,output_size=1024, num_layers=1, num_heads=4, dim_feedforward=256, dropout=0.25)
# t1 = joblib.load('transformer_1/orel/pretrainedModels/models/10Tokens/general_model.pkl')
t1 = joblib.load('transformer_1/orel/pretrainedModels/models/15Tokens\model_wiki_10414_36000.pkl')
# t1 = joblib.load('C:\\Users\\talia\\PycharmProjects\\HebrewLLM\\transformer_1\\orel\\pretrainedModels\\models\\10Tokens'
#                  '\\general_model.pkl')

# LLM model
llm_model_name = "facebook/opt-350m"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm = OPTForCausalLM.from_pretrained(llm_model_name)

# Transformer 2
# t2 = joblib.load('transformer_2/model_name.pkl')
t2 = HiddenStateTransformer2(input_size=512,output_size=512, num_layers=1, num_heads=2, dim_feedforward=256, dropout=0.15)
# t2 = joblib.load('C:\\Users\\orelz\\OneDrive\\שולחן העבודה\\work\\Ariel\\HebrewLLM\\transformer_2\\pretranedModels\\models\\15Tokens\\model_15_tokens_talia.pkl')
# t2 = joblib.load(
#     'C:\\Users\\talia\\PycharmProjects\\HebrewLLM\\transformer_2\\pretranedModels\\models\\15Tokens'
#     '\\model_15_tokens_talia.pkl')

# English to Hebrew translator
En_He_model_name = "Helsinki-NLP/opus-mt-en-he"
En_He_tokenizer = MarianTokenizer.from_pretrained(En_He_model_name)
En_He_translator_model = MarianMTModel.from_pretrained(En_He_model_name)

combined_model = CombinedModel(tokenizer1=He_En_tokenizer,
                               translator1=He_En_translator_model,
                               transformer1=t1,
                               llm_tokenizer=llm_tokenizer,
                               llm=llm,
                               transformer2=t2,
                               tokenizer2=En_He_tokenizer,
                               translator2=En_He_translator_model
                               )

# # for name, param in combined_model.named_parameters():
# #     if param.requires_grad:
# #         print(f'{name} requires grad')




# optimizer = optim.Adam(filter(lambda p: p.requires_grad, combined_model.parameters()), lr=lr)
optimizer = optim.Adam(combined_model.parameters(), lr=lr)

# optimizer = optim.SGD(filter(lambda p: p.requires_grad, combined_model.parameters()), lr=lr)

criterion = nn.CrossEntropyLoss()
# criterion = my_cross_entropy

# path = 'transformer_1/orel/wikipedia_data_15.csv'
path = 'transformer_1/orel/sampled_data.csv'
# path = 'C:\\Users\\talia\\PycharmProjects\\HebrewLLM\\wikipedia_data.csv'



train_combined_model(path,
                     train_size,
                     combined_model,
                     He_En_translator_model,
                     En_He_translator_model,
                     He_En_tokenizer,
                     En_He_tokenizer,
                     criterion,
                     optimizer,
                     5)

print(f"lr = {lr}")

# # for name, param in combined_model.named_parameters():
# #     if param.requires_grad:
# #         print(f'{name} requires grad')

# # save_model(combined_model, f'transformer_1/orel/pretrainedModels/models/combined/model_sampled_wiki_{train_size}_old_old.pkl')
# # save_model(combined_model, f'transformer_1/orel/pretrainedModels/models/combined/model_sampled_wiki_{train_size}_new_old.pkl')
# # save_model(combined_model, f'transformer_1/orel/pretrainedModels/models/combined/model_sampled_wiki_{train_size}_old_none.pkl')
# # save_model(combined_model, f'transformer_1/orel/pretrainedModels/models/combined/model_sampled_wiki_{train_size}_new_none.pkl')
# save_model(combined_model, f'transformer_1/orel/pretrainedModels/models/combined/model_sampled_wiki_{train_size}_none_none.pkl')

save_model(combined_model, f'transformer_1/orel/pretrainedModels/models/combined/model_sampled_wiki_{train_size}_new_none_2words_learning.pkl')
