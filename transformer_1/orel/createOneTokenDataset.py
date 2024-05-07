import pandas as pd
import numpy as np
import torch
from transformers import MarianTokenizer,MarianMTModel, AutoTokenizer, OPTForCausalLM
import csv


device = "cuda" if torch.cuda.is_available() else "cpu"

def create_empty_csv(path: str, headers = ['statement', 'translation']):
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()


def write_data_to_csv(data, headers = ['statement', 'translation'], path = "resources/big_one_token_dataset.csv"):
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        for statement, translation in data.items():
            writer.writerow({'statement': statement, 'translation': translation})


def create_one_token_words_data(fromSong: int, toSong:int, df: pd.DataFrame, hebrew_english_dict = {}):
    # Hebrew to english translator
    translator_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"

    translator_tokenizer = MarianTokenizer.from_pretrained(translator_model_name)
    translator_model = MarianMTModel.from_pretrained(translator_model_name)

    # OPT model
    OPT_model_name = "facebook/opt-350m"
    OPT_tokenizer = AutoTokenizer.from_pretrained(OPT_model_name)
    
    for i in range(fromSong,toSong):
        # ============= Get text to translate ============= 
        # Extract the 'songs' column as a list
        songs_list = df['songs'].iloc[i].strip('][').split(', ')
        # print(songs_list)

        # Remove single quotes from each element in the list
        hebrew_song_words = [song.strip("'") for song in songs_list]
        # print(hebrew_song_words)
        
        for hebrew_word in hebrew_song_words:
                
            if " " not in hebrew_word and hebrew_word not in hebrew_english_dict:
                # Translator
                inputs = translator_tokenizer(hebrew_word, return_tensors="pt")
                
                # Check if we have only 1 token exapt start token.
                if inputs.input_ids.size(1) == 2:
                
                    # Encode the source text
                    generated_ids = translator_model.generate(inputs.input_ids)
                    # print(generated_ids.size(1))

                    english_text = translator_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    
                    # OPT
                    opt_inputs = OPT_tokenizer(english_text, return_tensors="pt")
                    
                    if opt_inputs.input_ids.size(1) == 2:
                        hebrew_english_dict[hebrew_word] = english_text
        if (i - fromSong)% 10 == 0:
            print(f"{i - fromSong}/{toSong - fromSong} that is: {i/toSong - fromSong}% done.")
            write_data_to_csv(hebrew_english_dict)


def create_dataset(fromSong: int, toSong:int, songsDataPath = 'resources/HeSongsWords.csv', existing_dataset_path = "resources/big_one_token_dataset.csv"):
    
    df = pd.read_csv(songsDataPath)
    df = df.dropna()
    
    existing_dataset_df = pd.read_csv(existing_dataset_path)
    # Convert DataFrame to a dictionary with the first column as keys and the second column as values
    data_dict = pd.Series(existing_dataset_df.iloc[:, 1].values, index=existing_dataset_df.iloc[:, 0]).to_dict()

    if toSong - fromSong > 0:
        if (toSong > len(df['songs'])):
            toSong = len(df['songs'])
        if fromSong < 0:
            fromSong = 0
        
        create_one_token_words_data(fromSong, toSong, df, data_dict)





# create_dataset(1470)
a = {}
a['b'] = 'c'
a['d'] = 'e'

# create_empty_csv("resources/big_one_token_dataset.csv")
# append_data_to_csv("resources/big_one_token_dataset.csv",a)
create_dataset(0,30)


