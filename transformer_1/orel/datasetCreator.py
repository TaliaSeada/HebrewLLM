import pandas as pd
import numpy as np
import torch
from transformers import MarianTokenizer,MarianMTModel, AutoTokenizer, OPTForCausalLM,AutoModel
import csv
import time
import os
import h5py

MIN_SONG_NUMBER = 0
MAX_SONG_NUMBER = 1470
DISIRED_TOKEN_NUMBER = 10
SAVE_EVERY = 4

device = "cuda" if torch.cuda.is_available() else "cpu"


def append_to_pt_file(new_data: dict, i: int, file_path: str, log_file_path: str):


    # Save updated data back to the file
    torch.save(new_data, file_path)

    # Log the saving
    log_used_songs(fromSong= (i - SAVE_EVERY), toSong= i, log_file_path=log_file_path)
    print(f"Saved new data successfully.")
    

def log_used_songs(fromSong: int, toSong:int, log_file_path: str):
    message = f"Song {fromSong} - {toSong} were used.\n"
    
    print(message)
    
    with open(log_file_path, 'a') as file:
        file.write(message)  # Append the log entry to the file


def create_one_token_dataset(fromSong: int, toSong:int, df: pd.DataFrame, dataset_path: str, log_file_path: str, hebrew_english_dict = {}):
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
    for i in range(fromSong,toSong):
        
        new_words_hs = {}
        
        # ============= Get text to translate ============= 
        # Extract the 'songs' column as a list
        songs_list = df['songs'].iloc[i].strip('][').split(', ')

        # Remove single quotes from each element in the list
        hebrew_song_words = [song.strip("'") for song in songs_list]
        
        for hebrew_word in hebrew_song_words:
                
            if " " not in hebrew_word and hebrew_word not in hebrew_english_dict:
                # Translator
                inputs = translator_tokenizer(hebrew_word, return_tensors="pt")
                
                # Encode the source text
                generated_ids = translator_model.generate(inputs.input_ids)

                # Translation to english
                english_text = translator_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # OPT
                opt_inputs = OPT_tokenizer(english_text, return_tensors="pt", add_special_tokens=True)
                
                # Check if we have only 1 token exapt start token in the translator last hidden state & OPT first hidden state.
                
                if opt_inputs.input_ids.size(1) == DISIRED_TOKEN_NUMBER and generated_ids.size(1) == DISIRED_TOKEN_NUMBER + 1:

                    
                    # Append hidden states
                    translator_outputs = translator_model(input_ids=inputs.input_ids, decoder_input_ids=generated_ids, output_hidden_states=True)
                    
                    # # decoder_input_ids = opt_inputs.input_ids[:, 1:]  # This removes the first token, usually a start token
                    # decoder_input_ids = torch.cat([opt_inputs.input_ids, torch.tensor([[translator_tokenizer.eos_token_id]]).to(opt_inputs.input_ids.device)], dim=1)  # Append EOS token

                    # opt_outputs = opt_model(input_ids=opt_inputs.input_ids, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
                    opt_outputs = opt_model(input_ids=opt_inputs.input_ids, output_hidden_states=True)

                    # Extract the last hidden state from translator
                    translator_last_hidden_state = translator_outputs.decoder_hidden_states[-1]
                    
                    # Extract the first hidden state from OPT
                    opt_first_hidden_state = opt_outputs.hidden_states[1]
                    
                    # # Append in this format {hebrew: (english, hs1, hs2)}
                    # new_words_hs[hebrew_word] = (english_text, translator_last_hidden_state,opt_first_hidden_state)
                    
                    # Append words
                    hebrew_english_dict[i] = (translator_last_hidden_state,opt_first_hidden_state)
                    
                    # print(f"h = {hebrew_word}, E = {english_text}, trns = {translator_last_hidden_state.size(1)}, opt = {opt_first_hidden_state.size(1)}")
                    
        # Every 10 songs append new data to esisting
        if (i - fromSong)% SAVE_EVERY == 0:
            
            # Current time
            curr_time = time.time()
            
            print(f"{i - fromSong}/{toSong - fromSong} that is: {(i - fromSong)/(toSong - fromSong)*100}% done, running time = {curr_time - start_time} sec.")

            # Append the new words to csv
            if i > fromSong:
                # data1 = {'tensor1': ("abc", torch.randn(10, 256), torch.randn(10, 256))}
                append_to_pt_file(hebrew_english_dict, i, dataset_path, log_file_path)

                # Clear the dict, cause we dont want dups
                new_words_hs.clear()


def create_up_to_fixed_token_dataset(fromSong: int, toSong:int, df: pd.DataFrame, dataset_path: str, log_file_path: str, hebrew_english_dict = {}, desired_token_number = DISIRED_TOKEN_NUMBER):
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
    for i in range(fromSong,toSong):
        
        new_words_hs = {}
        
        # ============= Get text to translate ============= 
        # Extract the 'songs' column as a list
        songs_list = df['songs'].iloc[i].strip('][').split(', ')

        # Remove single quotes from each element in the list
        hebrew_song_words = [song.strip("'") for song in songs_list]
        
        text_size = i%desired_token_number
        counter = 0
        text = ""
        
        for j, hebrew_word in enumerate(hebrew_song_words):
            if counter < text_size:
                if counter == 0:
                    text += hebrew_word
                else:
                    text += " " + hebrew_word
                counter += 1
                continue
        
            
            # Translator
            inputs = translator_tokenizer(text, return_tensors="pt")
            
            # Encode the source text
            generated_ids = translator_model.generate(inputs.input_ids)

            # Translation to english
            english_text = translator_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # OPT
            opt_inputs = OPT_tokenizer(english_text, return_tensors="pt", add_special_tokens=True)

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
        if (i - fromSong)% SAVE_EVERY == 0:
            
            # Current time
            curr_time = time.time()
            
            print(f"{i - fromSong}/{toSong - fromSong} that is: {(i - fromSong)/(toSong - fromSong)*100}% done, running time = {curr_time - start_time} sec.")

            # Append the new words to csv
            if i > fromSong:
                # data1 = {'tensor1': ("abc", torch.randn(10, 256), torch.randn(10, 256))}
                append_to_pt_file(hebrew_english_dict, i, dataset_path, log_file_path)

                # Clear the dict, cause we dont want dups
                new_words_hs.clear()
                

def create_general_data(fromSong: int, toSong:int, df: pd.DataFrame, path_to_save: str, log_file_path: str, data_dict = {}):
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
    for i in range(fromSong,toSong):
        
        
        # ============= Get text to translate ============= 
        # Extract the 'songs' column as a list
        songs_list = df['songs'].iloc[i].strip('][').split(', ')

        # Remove single quotes from each element in the list
        hebrew_song_words = [song.strip("'") for song in songs_list]
        
        curr_song = " ".join(hebrew_song_words)
                        
        # Translator
        inputs = translator_tokenizer(curr_song, return_tensors="pt")
        
        # Encode the source text
        generated_ids = translator_model.generate(inputs.input_ids)

        # Translation to english
        english_text = translator_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # OPT
        opt_inputs = OPT_tokenizer(english_text, return_tensors="pt", add_special_tokens=True)
                
                    
        # Append hidden states
        translator_outputs = translator_model(input_ids=inputs.input_ids, decoder_input_ids=generated_ids, output_hidden_states=True)
        
        # # decoder_input_ids = opt_inputs.input_ids[:, 1:]  # This removes the first token, usually a start token
        # decoder_input_ids = torch.cat([opt_inputs.input_ids, torch.tensor([[translator_tokenizer.eos_token_id]]).to(opt_inputs.input_ids.device)], dim=1)  # Append EOS token

        # opt_outputs = opt_model(input_ids=opt_inputs.input_ids, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
        opt_outputs = opt_model(input_ids=opt_inputs.input_ids, output_hidden_states=True)

        # Extract the last hidden state from translator
        translator_last_hidden_state = translator_outputs.decoder_hidden_states[-1]
        
        # Extract the first hidden state from OPT
        opt_first_hidden_state = opt_outputs.hidden_states[1]
                    

        # Append words
        data_dict[i] = (translator_last_hidden_state,opt_first_hidden_state)
        
        #             # print(f"h = {hebrew_word}, E = {english_text}, trns = {translator_last_hidden_state.size(1)}, opt = {opt_first_hidden_state.size(1)}")
                    
        # Every 10 songs append new data to esisting
        if (i - fromSong)% SAVE_EVERY == 0:
            
            # Current time
            curr_time = time.time()
            
            print(f"{i - fromSong}/{toSong - fromSong} that is: {(i - fromSong)/(toSong - fromSong)*100}% done, running time = {curr_time - start_time} sec.")

            # Append the new words to csv
            if i > fromSong:
                # data1 = {'tensor1': ("abc", torch.randn(10, 256), torch.randn(10, 256))}
                append_to_pt_file(data_dict, i, path_to_save, log_file_path)


def create_dataset(fromSong: int, toSong:int, existing_dataset_path: str, log_file_path: str, songsDataPath = 'resources/HeSongsWords.csv', desired_token_num:int = -1):
    
    df = pd.read_csv(songsDataPath)
    df = df.dropna()
        
    # Load the words that already exists.
    data_dict = {}

    if os.path.exists(existing_dataset_path):
        data_dict = torch.load(existing_dataset_path)

    # Handle bad input
    if fromSong < MIN_SONG_NUMBER:
        fromSong = MIN_SONG_NUMBER
    if toSong > MAX_SONG_NUMBER:
        toSong = MAX_SONG_NUMBER
    
    if desired_token_num == -1:
        create_general_data(fromSong, toSong, df, existing_dataset_path, log_file_path, data_dict)
    else:
        # Create/Extend the dataset
        create_up_to_fixed_token_dataset(fromSong, toSong, df, existing_dataset_path, log_file_path, data_dict, desired_token_num)



# # create_empty_csv("resources/try.csv")
# # append_data_to_csv("resources/big_one_token_dataset.csv",a)
# create_dataset(100,201)


# # Initial data
# data1 = {'tensor1': ("abc", torch.randn(10, 256), torch.randn(10, 256))}

# # Save initial data (simulating first run)
# append_to_pt_file('resources/data.pt', data1)

# # New data to append
# data2 = {'tensor2': ("def", torch.randn(10, 256), torch.randn(10, 256))}

# # Append new data (simulating a subsequent run)
# append_to_pt_file('resources/data.pt', data2)

# Load and check contents
loaded_data = torch.load('resources/up_to_ten_tokens_dataset.pt')
# print(len(loaded_data.keys()))  # Should show both 'tensor1' and 'tensor2'
for key, (hs1, hs2) in loaded_data.items():
    print(key,hs1.shape, hs2.shape)


# create_dataset(8,50,"resources/up_to_ten_tokens_dataset.pt","resources/logs/used_songs_ten_tokens_dataset.log", desired_token_num=8)