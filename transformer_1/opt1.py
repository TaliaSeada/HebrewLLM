from transformers import AutoTokenizer, OPTForCausalLM
import pandas as pd
import numpy as np
from typing import Dict

model_to_use = "350m" #"6.7b" "2.7b" "1.3b" "350m" "125m"
layers_to_use = [0]
list_of_datasets = ["small_cities"] #[, "generated", "inventions", "elements", "animals", "facts", "companies"]

remove_period = True
model = OPTForCausalLM.from_pretrained("facebook/opt-"+model_to_use)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-"+model_to_use)

dfs: Dict[int, pd.DataFrame] = {}

for dataset_to_use in list_of_datasets:
    # Read the CSV file
    file = "C:\\Users\\talia\\PycharmProjects\\TranslatorGPT\\output_trans\\translate_short_words_1.csv"
    # file = "C:\\Users\\talia\\PycharmProjects\\TranslatorGPT\\resources\\" + dataset_to_use + "_heb_true_false.csv"
    df = pd.read_csv(file)
    df['embeddings'] = pd.Series(dtype='object')
    df['next_id'] = pd.Series(dtype=float)
    df['start'] = pd.Series(dtype='object')
    for layer in layers_to_use:
        dfs[layer] = df.copy()

    for i, row in df.iterrows():
        prompt = row['translation']
        if remove_period:
            prompt = prompt.rstrip(". ")
        inputs = tokenizer(prompt, return_tensors="pt")

        outputs = model.generate(inputs.input_ids, output_hidden_states=True, return_dict_in_generate=True, max_new_tokens=1, min_new_tokens=1)
        #, max_new_tokens=5, min_new_tokens=1) # return_logits=True, max_length=5, min_length=5, do_sample=True, temperature=0.5, no_repeat_ngram_size=3, top_p=0.92, top_k=10)return_logits=True
        # print()
        generate_ids = outputs[0]
        next_id = np.array(generate_ids)[0][-1]
        for layer in layers_to_use:
            last_hidden_state = outputs.hidden_states[0][layer][0][-1]
            start = outputs.hidden_states[0][layer][0][0]
            #[first_generated_word][layer][batch][input_words_for_first_generated_word_only]#last hidden state of first generated word
            dfs[layer].at[i, 'embeddings'] = [last_hidden_state.numpy().tolist()]
            dfs[layer].at[i, 'next_id'] = next_id
            dfs[layer].at[i, 'start'] = [start.numpy().tolist()]
            print("processing: " + str(i) + ", next_token:" + str(next_id))

    for layer in layers_to_use:
        dfs[layer].to_csv("C:\\Users\\talia\\PycharmProjects\\TranslatorGPT\\output\\short_words_opt_1.csv", index=False)



