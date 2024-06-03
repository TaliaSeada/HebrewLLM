# import pandas as pd

# from transformers import AutoTokenizer, AutoModel, MarianTokenizer,MarianMTModel, AutoTokenizer, OPTForCausalLM, AutoModelForSeq2SeqLM


# # English to Hebrew translator
# En_He_translator_name = "Helsinki-NLP/opus-mt-en-he"
# En_He_translator_model = AutoModelForSeq2SeqLM.from_pretrained(En_He_translator_name)
# En_He_tokenizer = AutoTokenizer.from_pretrained(En_He_translator_name)
    
# # Hebrew to english translator
# # He_En_translator_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
# # He_En_tokenizer = MarianTokenizer.from_pretrained(He_En_translator_name)
# # He_En_translator_model = MarianMTModel.from_pretrained(He_En_translator_name)


# # df = pd.read_csv('wikipedia_data.csv')

# # # print(type(df['Hebrew sentence']))


# # i = 0
# # for index, row in df.iterrows():
# #     if i > 1:
# #         break
#     # print(f"index: {index}, row: {row}")
    

#     # i += 1
# target_tokens = En_He_tokenizer("ירושלים היא עיר הבירה של", return_tensors="pt").input_ids
# print("Target Tokens:", target_tokens)

# # Decode each token ID to its corresponding word or subword
# token_ids = target_tokens.squeeze().tolist()
# decoded_tokens = [En_He_tokenizer.decode(token_id, skip_special_tokens=True) for token_id in token_ids]

# print("Decoded Tokens:", decoded_tokens)


# target_tokens = En_He_tokenizer("Jerusalem is the capital city of israel", return_tensors="pt").input_ids
# print("Target Tokens:", target_tokens)

# # Decode each token ID to its corresponding word or subword
# token_ids = target_tokens.squeeze().tolist()
# decoded_tokens = [En_He_tokenizer.decode(token_id, skip_special_tokens=True) for token_id in token_ids]

# print("Decoded Tokens:", decoded_tokens)

# # Encode the source text
# generated_ids = En_He_translator_model.generate(target_tokens)
# token_ids = generated_ids[0].squeeze().tolist()

# # Translation to hebrew
# decoded_tokens = [En_He_tokenizer.decode(token_id, skip_special_tokens=True) for token_id in token_ids]
# print("Decoded Tokens to hebrew:", decoded_tokens)


import torch
# Load dataset
loaded_data = torch.load("resources/datasets/dataset_wiki_up_to_15_tokens.pt")

print(loaded_data.items()[0])