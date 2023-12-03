import pandas as pd
from typing import Dict
from transformers import AutoModelForSeq2SeqLM
import os
from transformers import AutoTokenizer, OPTForCausalLM

os.environ['CUDA_VISIBLE_DEVICES'] = ''

layers_to_use = [-1]
list_of_datasets = ["cities"] #[, "generated",  "inventions", "elements", "animals", "facts", "companies"]

# Hebrew to English
remove_period = True
model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
device = "cpu"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dfs: Dict[int, pd.DataFrame] = {}


for dataset_to_use in list_of_datasets:
    # Read the CSV file
    file = "resources\\" + dataset_to_use + "_heb_true_false.csv"
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
        translated_tokens = model.generate(tokenized_sentence, output_hidden_states=True, return_dict_in_generate=True, max_length=512)
        translated_sentence = tokenizer.decode(translated_tokens[0][0], skip_special_tokens=True)

        for layer in layers_to_use:
            last_hidden_state = translated_tokens.encoder_hidden_states[layer][0][-1]
            dfs[layer].at[i, 'embeddings'] = [last_hidden_state.numpy().tolist()]
            dfs[layer].at[i, 'translation'] = translated_sentence
            print("processing: " + str(i) + ", translation:" + translated_sentence)

    for layer in layers_to_use:
        dfs[layer].to_csv("output_trans\\" + "translate_" + dataset_to_use + "_" + str(abs(layer)) + "_heb_rmv_period.csv", index=False)