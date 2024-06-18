from transformers import MarianTokenizer, MarianMTModel, AutoTokenizer, OPTForCausalLM

# Define the model names
model_name_he_en = "Helsinki-NLP/opus-mt-tc-big-he-en"
model_name_en_he = "Helsinki-NLP/opus-mt-en-he"

# Initialize the tokenizer
tokenizer: MarianTokenizer = MarianTokenizer.from_pretrained(model_name_he_en)
tokenizer2: MarianTokenizer = MarianTokenizer.from_pretrained(model_name_en_he)

# Load the models
model_he_en = MarianMTModel.from_pretrained(model_name_he_en)
model_en_he = MarianMTModel.from_pretrained(model_name_en_he)

# Example sentences
sentence_he = "שלום"
sentence_en = "Hello"


import torch

# English sentence
input_sentence = "at home today"

print("================= English to Hebrew =================")
# ================= English to Hebrew =================
# Tokenize the English sentence
input_tokens = tokenizer2(input_sentence, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')



print(f"tokens for english = {input_tokens.input_ids}")
ids = input_tokens
# ids.input_ids[0, 0] = 277
# ids.input_ids[0, 1] = 277
# ids.input_ids[0, 2] = 0
# ids.input_ids[0, 3] = 0


for t in input_tokens.input_ids:
    print(f"token {t} = {tokenizer2.convert_ids_to_tokens(t)}")


# Generate translation in Hebrew
with torch.no_grad():
    tensor = torch.tensor([227, 366, 0])
    translated_tokens = model_en_he.generate(**ids)
    print(f"Translation to hebrew tokens = {translated_tokens[0]}")
    
print(f"Without first token = {translated_tokens[0][1:]}")
    
translated_sentence = tokenizer2.decode(translated_tokens[0], skip_special_tokens=True)

for t in translated_tokens:
    print(f"translated token {t} = {tokenizer2.convert_ids_to_tokens(t)}")

print("Translated Sentence:", translated_sentence)


# print(tokenizer2.convert_tokens_to_ids("home"))
# print(tokenizer2.convert_tokens_to_ids("הביתה"))
# print(tokenizer2.convert_tokens_to_ids("הולך"))
# print(tokenizer2.convert_ids_to_tokens(277))
# print(tokenizer2.convert_ids_to_tokens(13085))
# print(tokenizer2.convert_ids_to_tokens(34751))
# print(tokenizer2.convert_ids_to_tokens(1))
# print(tokenizer2.convert_ids_to_tokens(29275))


# print(tokenizer2.current_encoder)
print(tokenizer2.convert_tokens_to_ids("there wiil be"))


# Set the tokenizer as the target tokenizer for decoding
with tokenizer2.as_target_tokenizer():
    ts = tokenizer2.encode("ולכשתבין את דברי", skip_special_tokens=True)

# print(f"Translated text: {translated_text}")

print(f"translated tokens {ts} = {[tokenizer2.convert_ids_to_tokens(t) for t in ts]}")



# print(tokenizer2.add)
# # ================= English to Hebrew =================



# # ================= Hebrew to English =================
# print("================= Hebrew to English =================")
# # Tokenize the English sentence
# input_tokens = tokenizer(input_sentence, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')

# print(f"tokens for Hebrew input = {input_tokens.input_ids}")

# for t in input_tokens.input_ids:
#     print(f"token {t} = {tokenizer.convert_ids_to_tokens(t)}")

# # Generate translation in Hebrew
# with torch.no_grad():
#     translated_tokens = model_he_en.generate(**input_tokens)
#     print(f"Translation to English tokens = {translated_tokens[0]}")
# translated_sentence = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

# for t in translated_tokens:
#     print(f"translated token {t} = {tokenizer.convert_ids_to_tokens(t)}")

# print("Translated Sentence:", translated_sentence)
# # ================= Hebrew to English =================



# from transformers import MarianTokenizer
# import sentencepiece as spm

# # Initialize the Marian tokenizer for English to Hebrew translation
# model_name_en_he = "Helsinki-NLP/opus-mt-en-he"
# tokenizer_en_he: MarianTokenizer = MarianTokenizer.from_pretrained(model_name_en_he)


# tokenizer_en_he.current_spm


# hebrew_sentence = "ירושלים היא עיר הבירה של ישראל"
# english_sentence = "Jerusalem is the capital of Israel"
# # Access the SentencePiece model
# sp = tokenizer2.current_spm

# input_tokens = tokenizer2(hebrew_sentence)

# print(f"before tokens mixed input = {input_tokens.input_ids}")

# dic = tokenizer2.tokenize(hebrew_sentence)
# print(f"dic = {dic}")

# # tokenizer2.

# enc = tokenizer2.spm_target.Encode(hebrew_sentence)
# print(f"enc = {enc}")
# encEn = tokenizer2.spm_source.Encode(hebrew_sentence)
# print(f"English enc = {encEn}")
# dec = tokenizer2.decode(enc)
# print(f"dec = {dec}")
# decEn = tokenizer2.decode(encEn)
# print(f"English dec = {decEn}")


# print(dic)
# # tokenizer2.
# input_tokens = tokenizer2(hebrew_sentence)


# print(f"After tokens mixed input = {input_tokens.input_ids}")

# Hebrew sentence

# # Tokenize the Hebrew sentence using SentencePiece
# tokens = sp.encode_as_pieces(hebrew_sentence)
# token_ids = sp.encode_as_ids(hebrew_sentence)

# print("Tokens:", tokens)
# print("Token IDs:", token_ids)

# # Tokenize the Hebrew sentence using SentencePiece
# tokens = sp.encode_as_pieces(input_sentence)
# token_ids = sp.encode_as_ids(input_sentence)

# print("Tokens:", tokens)
# print("Token IDs:", token_ids)










# # Tokenize the translated Hebrew sentence
# hebrew_tokens = tokenizer_en_he(translated_sentence, return_tensors="pt")

# print(f"Actual translated sentence (hebrew input) tokens = {hebrew_tokens.input_ids}")
# # Decode each token ID to its corresponding word or subword
# token_ids = hebrew_tokens.input_ids.squeeze().tolist()
# decoded_tokens = [tokenizer_en_he.decode(token_id, skip_special_tokens=True) for token_id in token_ids]

# print("Hebrew Tokens (input_ids):", hebrew_tokens.input_ids)

# print("Decoded Tokens:", decoded_tokens)

# Get the vocabulary dictionary: token -> token ID
# vocab = tokenizer_en_he.get_vocab()
# print(tokenizer_en_he.convert_ids_to_tokens(48))
# print(tokenizer_en_he.convert_tokens_to_ids("ה"))
# Invert the vocabulary dictionary: token ID -> token
# id_to_token = {v: k for k, v in vocab.items()}
# print(vocab.items())
