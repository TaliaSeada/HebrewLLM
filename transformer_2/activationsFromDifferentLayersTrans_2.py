import pandas as pd
from typing import Dict

import torch
from transformers import AutoModelForSeq2SeqLM
import os
from transformers import AutoTokenizer, OPTForCausalLM

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
        file = "C:\\Users\\talia\\PycharmProjects\\TranslatorGPT\\resources\\" + dataset_to_use + "_true_false.csv"
        df = pd.read_csv(file)
        df['embeddings'] = pd.Series(dtype='object')
        df['translation'] = pd.Series(dtype='object')
        for layer in layers_to_use:
            dfs[layer] = df.copy()

        for i, row in df.iterrows():
            prompt = row['statement']
            if remove_period:
                prompt = prompt.rstrip(". ")

            # sentences = nltk.sent_tokenize(prompt)
            # translated_sentences = []
            # for sentence in sentences:
            #     tokenized_sentence = tokenizer.encode(sentence, return_tensors='pt').to(device)
            #     translated_tokens = model.generate(tokenized_sentence, max_length=512)
            #     translated_sentence = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            #     translated_sentences.append(translated_sentence)
            #
            # translated_text = " ".join(translated_sentences)

            tokenized_sentence = tokenizer.encode(prompt, return_tensors='pt').to(device)
            translated_tokens = model.generate(tokenized_sentence, output_hidden_states=True,
                                               return_dict_in_generate=True, max_length=512)
            translated_sentence = tokenizer.decode(translated_tokens[0][0], skip_special_tokens=True)

            for layer in layers_to_use:
                last_hidden_state = translated_tokens.encoder_hidden_states[layer][0][-1]
                dfs[layer].at[i, 'embeddings'] = [last_hidden_state.numpy().tolist()]
                dfs[layer].at[i, 'translation'] = translated_sentence
                print("processing: " + str(i) + ", translation:" + translated_sentence)

        for layer in layers_to_use:
            dfs[layer].to_csv(
                "C:\\Users\\talia\\PycharmProjects\\TranslatorGPT\\output_trans\\" + "translate_with_labels_" + dataset_to_use + "_" + str(
                    abs(layer)) + "_rmv_period.csv", index=False)


def translate_sen(prompt, layer):
    remove_period = True
    model_name = "Helsinki-NLP/opus-mt-en-he"
    device = "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if remove_period:
        prompt = prompt.rstrip(". ")

    tokenized_sentence = tokenizer.encode(prompt, return_tensors='pt').to(device)
    translated_tokens = model.generate(tokenized_sentence, output_hidden_states=True, return_dict_in_generate=True,
                                       max_length=512)
    translated_sentence = tokenizer.decode(translated_tokens[0][0], skip_special_tokens=True)
    last_hidden_state = translated_tokens.encoder_hidden_states[layer][0][-1]
    embeddings = [last_hidden_state.numpy().tolist()]

    return translated_tokens, translated_sentence, embeddings


def translate_sen_from_embeddings(embeddings):
    model_name = "Helsinki-NLP/opus-mt-en-he"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Assuming the embeddings are in the shape (sequence_length, embedding_dim)
    # Convert embeddings to a PyTorch tensor and create a dummy input token tensor for the model
    embeddings_tensor = torch.tensor(embeddings)
    input_ids = torch.zeros((1, embeddings_tensor.shape[0]), dtype=torch.long)

    # Generate translations using the model
    translated_tokens = model.generate(input_ids, attention_mask=None,
                                       encoder_outputs=(None, None, embeddings_tensor.unsqueeze(0)))

    # Decode the translated tokens into a sentence
    translated_sentence = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    return translated_tokens, translated_sentence
if __name__ == '__main__':
    layers_to_use = [1]
    list_of_datasets = ["cities"] # ["generated", "inventions", "elements", "animals", "facts", "companies"]
    activateTrans2(layers_to_use, list_of_datasets)