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
opt_layer = -1

# translator english-hebrew
model_name_en = "Helsinki-NLP/opus-mt-en-he"
translator_model_en = AutoModelForSeq2SeqLM.from_pretrained(model_name_en)
translator_tokenizer_en = AutoTokenizer.from_pretrained(model_name_en)
translator_layer_en = 1

# # translator hebrew-english
# model_name_he = "Helsinki-NLP/opus-mt-tc-big-he-en"
# translator_model_he = AutoModelForSeq2SeqLM.from_pretrained(model_name_he)
# translator_tokenizer_he = AutoTokenizer.from_pretrained(model_name_he)
# translator_layer_he = -1

if __name__ == '__main__':
    # Load data using the link above
    # TODO change to the link, TEMPORARY
    data_path = 'C:\\Users\\talia\\PycharmProjects\\HebrewLLM\\resources\\dict.csv'
    df = pd.read_csv(data_path)
    df['translation'] = df['translation'].astype(str)

    # Prepare data
    data = []
    for i, row in df.iterrows():
        prompt = row['translation']


        # OPT last layer
        opt_inputs = opt_tokenizer(prompt, return_tensors="pt")
        opt_outputs = opt_model(**opt_inputs, output_hidden_states=True)
        opt_hidden_state = opt_outputs.hidden_states[opt_layer]

        # Translator first layer
        translator_inputs = translator_tokenizer_en(prompt, return_tensors="pt")
        decoder_start_token_id = translator_tokenizer_en.pad_token_id
        decoder_input_ids = torch.full((translator_inputs.input_ids.size(0), 1), decoder_start_token_id,
                                       dtype=torch.long)
        translator_outputs = translator_model_en(input_ids=translator_inputs.input_ids,
                                              decoder_input_ids=decoder_input_ids, output_hidden_states=True)
        translator_hidden_state = translator_outputs.encoder_hidden_states[translator_layer_en]


        # Filter out long words
        if opt_hidden_state.shape[1] != 2 or translator_hidden_state.shape[1] != 2:
            continue

        translator_outputs = translator_model_en.generate(input_ids=translator_inputs.input_ids,
                                                          decoder_input_ids=decoder_input_ids,
                                                          max_length=20,  # adjust max_length as needed
                                                          num_beams=5,  # adjust num_beams as needed
                                                          early_stopping=True,
                                                          output_hidden_states=True)
        translated_text = translator_tokenizer_en.decode(translator_outputs[0], skip_special_tokens=True)

        data.append((prompt, translated_text))
        print(data[-1])

    # Convert data to DataFrame
    df_final = pd.DataFrame(data, columns=['English', 'Hebrew'])

    # Save DataFrame to CSV
    csv_filename = 'translated_data.csv'
    df_final.to_csv(csv_filename, index=False)

    print("CSV file saved successfully as:", csv_filename)