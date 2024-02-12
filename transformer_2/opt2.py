import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, OPTForCausalLM,  AutoModelForSeq2SeqLM

data_path = 'C:\\Users\\talia\\PycharmProjects\\TranslatorGPT\\resources\\dict.csv'
df = pd.read_csv(data_path)
remove_period = True

# opt
model_to_use = "350m"
opt_model = OPTForCausalLM.from_pretrained("facebook/opt-" + model_to_use)
opt_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-" + model_to_use)
opt_layer = -1

# translator
model_name = "Helsinki-NLP/opus-mt-en-he"
translator_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
translator_tokenizer = AutoTokenizer.from_pretrained(model_name)
translator_layer = 0

for i, row in df.iterrows():
    prompt = row['translation']

    # OPT last layer
    opt_inputs = opt_tokenizer(prompt, return_tensors="pt")
    opt_outputs = opt_model(**opt_inputs, output_hidden_states=True)
    opt_hidden_state = opt_outputs.hidden_states[opt_layer]

    # translator first layer
    translator_inputs = translator_tokenizer(prompt, return_tensors="pt")
    decoder_start_token_id = translator_tokenizer.pad_token_id
    decoder_input_ids = torch.full((translator_inputs.input_ids.size(0), 1), decoder_start_token_id, dtype=torch.long).to(
        translator_inputs.input_ids.device)

    translator_outputs = translator_model(input_ids=translator_inputs.input_ids, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
    translator_hidden_state = translator_outputs.encoder_hidden_states[translator_layer]

    # we want for now only the short words
    if opt_hidden_state.shape[1] != 2 and translator_hidden_state.shape[1] != 2:
        continue


    print(i)