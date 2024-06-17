import torch
import torch.nn as nn
from transformers import AutoTokenizer, OPTForCausalLM, AutoModelForSeq2SeqLM
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

'''
TODO take words from here and make data set
https://www.kaggle.com/datasets/bcruise/wordle-valid-words
'''

# opt
model_to_use = "350m"
opt_model = OPTForCausalLM.from_pretrained("facebook/opt-" + model_to_use)
opt_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-" + model_to_use)
opt_layer = 1

# translator hebrew-english
model_name_he = "Helsinki-NLP/opus-mt-tc-big-he-en"
translator_model_he = AutoModelForSeq2SeqLM.from_pretrained(model_name_he)
translator_tokenizer_he = AutoTokenizer.from_pretrained(model_name_he)
translator_layer_he = -1

def remove_punctuation(text):
    punctuation = ",()[]'\""
    for char in punctuation:
        text = text.replace(char, '')
    return text


if __name__ == '__main__':
    # Load data using the link above
    # TODO change to the link, TEMPORARY
    # data_path = 'C:\\Users\\talia\\PycharmProjects\\HebrewLLM\\resources\\dict.csv'
    # data_path = 'C:\\Users\\talia\\PycharmProjects\\HebrewLLM\\resources\\valid_guesses.csv'
    data_path = 'resources/HeSongsWords.csv'

    df = pd.read_csv(data_path)
    df['song'] = df['song'].astype(str)
    print(df)

    # Prepare data
    data = []
    for i, row in df.iterrows():
        if i==100:
            break
        prompt = row['song']
        # print(prompt)

        # Translator last layer
        translator_inputs = translator_tokenizer_he(prompt, return_tensors="pt")
        decoder_start_token_id = translator_tokenizer_he.pad_token_id
        decoder_input_ids = torch.full((translator_inputs.input_ids.size(0), 1), decoder_start_token_id,
                                       dtype=torch.long)
        translator_outputs = translator_model_he(input_ids=translator_inputs.input_ids,
                                                 decoder_input_ids=decoder_input_ids, output_hidden_states=True)
        translator_hidden_state = translator_outputs.encoder_hidden_states[translator_layer_he]

        # Translator output, to use in the OPT
        translator_outputs = translator_model_he.generate(input_ids=translator_inputs.input_ids,
                                                          decoder_input_ids=decoder_input_ids,
                                                          max_length=16,  # adjust max_length as needed
                                                          num_beams=5,  # adjust num_beams as needed
                                                          early_stopping=True,
                                                          output_hidden_states=True)
        translated_text = translator_tokenizer_he.decode(translator_outputs[0], skip_special_tokens=True)

        data.append((prompt, translated_text))
        print(data[-1])

    # Convert data to DataFrame
    df_final = pd.DataFrame(data, columns=['statement', 'translation'])

    # Save DataFrame to CSV
    csv_filename = 'He_En_Data.csv'
    df_final.to_csv(csv_filename, index=False)

    print("CSV file saved successfully as:", csv_filename)