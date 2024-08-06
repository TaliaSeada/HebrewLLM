import pandas as pd
from transformers import AutoTokenizer, AutoModel, MarianTokenizer, MarianMTModel, AutoTokenizer, OPTForCausalLM
import torch

df = pd.read_csv("wikipedia_data.csv")
df = df.dropna()


# Hebrew to english translator
He_En_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
He_En_tokenizer = MarianTokenizer.from_pretrained(He_En_model_name)
He_En_translator_model = MarianMTModel.from_pretrained(He_En_model_name)

new_data = []

for index, row in df.iterrows():
    
    target_hebrew_sentence = row['Hebrew sentence'] + " " + row['label']
    
    # Translator
    inputs = He_En_tokenizer(target_hebrew_sentence, return_tensors="pt")

    # Encode the source text
    generated_ids = He_En_translator_model.generate(inputs.input_ids)
    
    tokens_num = len(generated_ids[0])

    if tokens_num < 15:
        # append data frame to CSV file
        # df.to_csv('wikipedia_data_15.csv', mode='a', index=False, header=True)
        new_data.append(row)
    else:
        pass # Save the test data that we didn't use here (about 6K rows out of 36K)

    if index % 100 == 0:
        
        # Create the new DataFrame from the modified data
        df_target = pd.DataFrame(new_data)
        df_target.to_csv('wikipedia_data_15.csv', mode='a', index=False, header=True)
        new_data.clear()
        
        print(f"Just finished {index} sentences")

