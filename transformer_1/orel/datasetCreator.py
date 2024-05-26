import pandas as pd
import numpy as np
import torch
from transformers import MarianTokenizer,MarianMTModel, AutoTokenizer, OPTForCausalLM,AutoModel
import csv
import time
import os
import h5py

MIN_SENTENCE_NUMBER = 0
MAX_SENTENCE_NUMBER = 1470
DISIRED_TOKEN_NUMBER = 15
SAVE_EVERY = 4

device = "cuda" if torch.cuda.is_available() else "cpu"


def append_to_pt_file(new_data: dict, i: int, file_path: str, log_file_path: str):


    # Save updated data back to the file
    torch.save(new_data, file_path)

    # Log the saving
    log_used_songs(fromIndex= (i - SAVE_EVERY), toIndex= i, log_file_path=log_file_path)
    print(f"Saved new data successfully.")
    

def log_used_songs(fromIndex: int, toIndex:int, log_file_path: str):
    message = f"Sentence {fromIndex} - {toIndex} were used.\n"
    
    print(message)
    
    with open(log_file_path, 'a') as file:
        file.write(message)  # Append the log entry to the file


def createHebrewWordsArray(words_list: list):

    # Remove single quotes from each element in the list
    return [sentence.strip("'") for sentence in words_list]


def createHebrewWordsArrayFromSongs(df: pd.DataFrame, i: int):
    sentence_list = df['songs'].iloc[i].strip('][').split(', ')
    return createHebrewWordsArray(sentence_list)


def createHebrewWordsArrayFromWiki(df: pd.DataFrame, i: int):
    sentence_list = df['Hebrew sentence'].iloc[i].strip('][').split(' ')
    sentence_list.append(df['label'].iloc[i].strip(']['))
    # print(sentence_list)
    return createHebrewWordsArray(sentence_list)


def create_up_to_fixed_token_dataset(fromIndex: int, toIndex:int, df: pd.DataFrame, dataset_path: str, log_file_path: str, hebrew_english_dict = {}, desired_token_number = DISIRED_TOKEN_NUMBER, dataset_name = "songs"):
    # Hebrew to english translator
    translator_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"

    translator_tokenizer = MarianTokenizer.from_pretrained(translator_model_name)
    translator_model = MarianMTModel.from_pretrained(translator_model_name)

    # OPT model
    OPT_model_name = "facebook/opt-350m"
    OPT_tokenizer = AutoTokenizer.from_pretrained(OPT_model_name)
    opt_model = AutoModel.from_pretrained(OPT_model_name).to(device)
    
    # Start the timer
    start_time = time.time()
    
    # Loop over all the songs
    for i in range(fromIndex,toIndex):
        
        new_words_hs = {}
        
        # ============= Get text to translate ============= 
        # Remove single quotes from each element in the list
        hebrew_words = []
        
        if dataset_name == "songs":
            hebrew_words = createHebrewWordsArrayFromSongs(df, i)
        else:
            hebrew_words = createHebrewWordsArrayFromWiki(df, i)
        
        
        
        token_number = desired_token_number if len(hebrew_words) > desired_token_number and desired_token_number != -1 else len(hebrew_words) - 1

        # print(f"len(hebrew_words) = {len(hebrew_words)}, desired_token_number = {desired_token_number}, token_number = {token_number}")
        
        # text_size = i % token_number
        text_size = token_number
        counter = 0
        text = ""
        
        for j, hebrew_word in enumerate(hebrew_words):
            # print(f"hebrew_word = {hebrew_word}")
            if counter < text_size:
                if counter == 0:
                    text += hebrew_word
                else:
                    text += " " + hebrew_word
                counter += 1
                # print(f"text = {text}")
                continue
            
            print(f"total text = {text}")
            
        
            
            # Translator
            inputs = translator_tokenizer(text, return_tensors="pt")
            
            # Encode the source text
            generated_ids = translator_model.generate(inputs.input_ids)

            # Translation to english
            english_text = translator_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # OPT
            opt_inputs = OPT_tokenizer(english_text, return_tensors="pt", add_special_tokens=True)

            # print(f"IsSaving: {opt_inputs.input_ids.size(1) <= desired_token_number and generated_ids.size(1) <= desired_token_number + 1}")
            
            # print(f"opt_hs_size = {opt_inputs.input_ids.size(1)}, trns_ids_size = {generated_ids.size(1)}")
            
            # Check if we have only 1 token exapt start token in the translator last hidden state & OPT first hidden state.
            if opt_inputs.input_ids.size(1) <= desired_token_number and generated_ids.size(1) <= desired_token_number + 1:

                # Append hidden states
                translator_outputs = translator_model(input_ids=inputs.input_ids, decoder_input_ids=generated_ids, output_hidden_states=True)

                # opt_outputs = opt_model(input_ids=opt_inputs.input_ids, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
                opt_outputs = opt_model(input_ids=opt_inputs.input_ids, output_hidden_states=True)

                # Extract the last hidden state from translator
                translator_last_hidden_state = translator_outputs.decoder_hidden_states[-1]

                # Extract the first hidden state from OPT
                opt_first_hidden_state = opt_outputs.hidden_states[1]
                
                # Append words
                hebrew_english_dict[f"{i}_{j}"] = (translator_last_hidden_state,opt_first_hidden_state)

                # print(f"h = {hebrew_word}, E = {english_text}, trns = {translator_last_hidden_state.size(1)}, opt = {opt_first_hidden_state.size(1)}")

            
            counter = 0
            text = ""
        
        # Every 10 songs append new data to esisting
        if (i - fromIndex)% SAVE_EVERY == 0:
            
            # Current time
            curr_time = time.time()
            
            print(f"{i - fromIndex}/{toIndex - fromIndex} that is: {(i - fromIndex)/(toIndex - fromIndex)*100}% done, running time = {curr_time - start_time} sec.")

            # Append the new words to csv
            if i > fromIndex:
                # data1 = {'tensor1': ("abc", torch.randn(10, 256), torch.randn(10, 256))}
                append_to_pt_file(hebrew_english_dict, i, dataset_path, log_file_path)

                # Clear the dict, cause we dont want dups
                new_words_hs.clear()


def create_dataset(fromIndex: int, toIndex:int, existing_dataset_path: str, log_file_path: str, dataPath = 'resources/HeSongsWords.csv', desired_token_num:int = -1):
    
    df = pd.read_csv(dataPath)
    df = df.dropna()
    print(f"df.shape = {df.shape}")
        
    # Load the words that already exists.
    data_dict = {}

    if os.path.exists(existing_dataset_path):
        data_dict = torch.load(existing_dataset_path)
            
    dataset_name = "wiki" if dataPath == 'wikipedia_data.csv' else "songs"
    
    print(f"Starting to create dataset based on {dataset_name} data, from: {fromIndex}, to: {toIndex}")
    
    # Create/Extend the dataset
    create_up_to_fixed_token_dataset(fromIndex, toIndex, df, existing_dataset_path, log_file_path, data_dict, desired_token_num, dataset_name)



# # # create_dataset(8,50,"resources/datasets/up_to_ten_tokens_dataset.pt","resources/logs/used_songs_ten_tokens_dataset.log", desired_token_num=8)
# create_dataset(34020,36000,"resources/datasets/up_to_ten_tokens_dataset_wiki_5.pt","resources/logs/used_songs_ten_tokens_dataset_wiki_5.log", 'wikipedia_data.csv', desired_token_num=14)

# Load and check contents
loaded_data = torch.load('resources/datasets/up_to_ten_tokens_dataset_wiki_5.pt')
print(len(loaded_data.keys()))  # Should show both 'tensor1' and 'tensor2'

# for key, (hs1, hs2) in loaded_data.items():
#     print(key,hs1.shape, hs2.shape)