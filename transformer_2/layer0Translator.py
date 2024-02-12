import pandas as pd
from typing import Dict
import torch
from transformers import AutoModelForSeq2SeqLM
import os
from transformers import AutoTokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def activateTrans2(layers_to_use, list_of_datasets):
    # English to Hebrew
    remove_period = True
    model_name = "Helsinki-NLP/opus-mt-en-he"
    device = "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dfs: Dict[int, pd.DataFrame] = {}

    for dataset_to_use in list_of_datasets:
        # Read the CSV file
        file = "C:\\Users\\talia\\PycharmProjects\\TranslatorGPT\\resources\\" + dataset_to_use + ".csv"
        df = pd.read_csv(file)
        df['embeddings'] = pd.Series(dtype='object')
        for layer in layers_to_use:
            dfs[layer] = df.copy()

        for i, row in df.iterrows():
            prompt = row['translation']
            if remove_period:
                prompt = prompt.rstrip(". ")

            inputs = tokenizer(prompt, return_tensors="pt")
            decoder_start_token_id = tokenizer.pad_token_id
            decoder_input_ids = torch.full((inputs.input_ids.size(0), 1), decoder_start_token_id, dtype=torch.long).to(
                inputs.input_ids.device)

            outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_input_ids, output_hidden_states=True)


            for layer in layers_to_use:
                last_hidden_state = outputs.encoder_hidden_states[layer]
                dfs[layer].at[i, 'embeddings'] = [last_hidden_state.detach().numpy().tolist()]

        for layer in layers_to_use:
            dfs[layer].to_csv(
                "C:\\Users\\talia\\PycharmProjects\\TranslatorGPT\\output_trans\\" + dataset_to_use + ".csv", index=False)


if __name__ == '__main__':
    layers_to_use = [0]
    list_of_datasets = ['dict']
    activateTrans2(layers_to_use, list_of_datasets)