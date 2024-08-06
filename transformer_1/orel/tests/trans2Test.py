# import joblib
# import torch
# from combinedTransformersModel import CombinedModel
# from transformers import AutoTokenizer, AutoModel, MarianTokenizer, MarianMTModel, AutoTokenizer, OPTForCausalLM
# from generalTransformer import CustomLayerWrapper, CustomLayerWrapper2



# # English to Hebrew translator
# En_He_model_name = "Helsinki-NLP/opus-mt-en-he"
# En_He_tokenizer = MarianTokenizer.from_pretrained(En_He_model_name)
# En_He_translator_model = MarianMTModel.from_pretrained(En_He_model_name)

# # LLM model
# llm_model_name = "facebook/opt-350m"
# llm_tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(llm_model_name)
# llm:OPTForCausalLM = OPTForCausalLM.from_pretrained(llm_model_name)


# def generate_test_inputs(e_text):
#     llm_inputs = llm_tokenizer(e_text, return_tensors="pt")
    
#     llm_outputs = llm(input_ids=llm_inputs.input_ids, output_hidden_states=True)
    
#     # Extract the last hidden state from llm
#     return llm_outputs.hidden_states[-1]


# def generate_predicted_distribution(text):
    
#     # Get the tokens for the target sentence
#     known_target_ids = En_He_tokenizer(text_target=text, return_tensors="pt").input_ids
    
#     # Get the padding token ID
#     pad_token_id = En_He_tokenizer.pad_token_id

#     # Create a tensor filled with pad token IDs, with the desired length
#     max_length = 15
#     # batch_size = known_target_ids.size(0)
#     # decoder_input_ids = torch.full((batch_size, max_length - known_target_ids.shape[1]), pad_token_id, dtype=torch.long)

        
#     # # Trick Translator by giving it a dummy that contain the desired number of tokens (In our case 15)
#     # # and replace the first layer as it got other word embedding.
#     inputs = En_He_tokenizer("a " * 14, return_tensors="pt")
    
    
#     # Ensure the attention mask is correctly shaped
#     attention_mask = torch.ones((1, 15))
#     inputs['attention_mask'] = attention_mask
    
#     decoder_len = max((max_length - known_target_ids.shape[1]), 0)
    
#     # Prepare decoder_input_ids, starting with the <pad> token
#     decoder_input_ids = torch.full(
#         (inputs.input_ids.size(0),  decoder_len), En_He_tokenizer.pad_token_id, dtype=torch.long
#     )

#     # Concatenate with input_ids shifted right
#     decoder_input_ids = torch.cat([known_target_ids, decoder_input_ids], dim=1)

#     # print(f"decoder_input_ids = {decoder_input_ids},\nShape = {decoder_input_ids.shape}")

#     # Forward pass to get the logits
#     outputs = En_He_translator_model(
#         input_ids=inputs.input_ids,
#         attention_mask=attention_mask,
#         decoder_input_ids=decoder_input_ids,
#         output_hidden_states=True
#     )
        
#     # Access the generated token IDs
#     token_ids = outputs.logits.argmax(-1)

#     generated_text = En_He_translator_model.decode(token_ids[0], skip_special_tokens=True)
    
#     # Print the generated text
#     print("Generated Text: ", generated_text)
    
    
# e_text = "I"
# # e_text = "I'm"
# # e_text = "do"


# llm_last_hs = generate_test_inputs(e_text)

# t2 = joblib.load('transformer_2/pretrainedModels/models/15Tokens\model_15_tokens_talia.pkl')

# t2.eval()
# with torch.no_grad():

#     trans2_first_hs = t2(llm_last_hs)
    
#     print(trans2_first_hs.shape)

#     # Trick 
#     original_layer2 = En_He_translator_model.model.encoder.layers[1]
#     wrapped_layer2 = CustomLayerWrapper2(original_layer2, None)
#     En_He_translator_model.model.encoder.layers[1] = wrapped_layer2

#     # Inject embeddings to the first layer of the translator
#     En_He_translator_model.model.encoder.layers[1].hs = trans2_first_hs





#     # Get the tokens for the target sentence
#     known_target_ids = En_He_tokenizer(text_target=e_text, return_tensors="pt").input_ids
    
#     # Get the padding token ID
#     pad_token_id = En_He_tokenizer.pad_token_id

#     # Create a tensor filled with pad token IDs, with the desired length
#     max_length = 15
#     # batch_size = known_target_ids.size(0)
#     # decoder_input_ids = torch.full((batch_size, max_length - known_target_ids.shape[1]), pad_token_id, dtype=torch.long)

        
#     # # Trick Translator by giving it a dummy that contain the desired number of tokens (In our case 15)
#     # # and replace the first layer as it got other word embedding.
    
#     # inputs = En_He_tokenizer("a " * 14, return_tensors="pt")
#     inputs = En_He_tokenizer("a " * 14, return_tensors="pt")
    
#     print(len(inputs))
    
#     # Ensure the attention mask is correctly shaped
#     attention_mask = torch.ones((1, 15))
#     inputs['attention_mask'] = attention_mask
    
#     decoder_len = max((max_length - known_target_ids.shape[1]), 0)

#     # Prepare decoder_input_ids, starting with the <pad> token
#     decoder_input_ids = torch.full(
#         (inputs.input_ids.size(0),  decoder_len), En_He_tokenizer.pad_token_id, dtype=torch.long
#     )

#     # Concatenate with input_ids shifted right
#     decoder_input_ids = torch.cat([known_target_ids, decoder_input_ids], dim=1)


    
#     print(f"decoder_input_ids = {decoder_input_ids},\nShape = {decoder_input_ids.shape}")

#     # Forward pass to get the logits
#     outputs = En_He_translator_model(
#         input_ids=inputs.input_ids,
#         attention_mask=attention_mask,
#         decoder_input_ids=decoder_input_ids,
#         output_hidden_states=True
#     )

#     # Access the generated token IDs
#     token_ids = outputs.logits.argmax(-1)

#     generated_text = En_He_translator_model.decode(token_ids[0], skip_special_tokens=True)

#     # Print the generated text
#     print("Generated Text: ", generated_text)