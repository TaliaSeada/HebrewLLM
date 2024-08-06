

import joblib
import pandas as pd

# from transformers import AutoTokenizer, AutoModel, MarianTokenizer, MarianMTModel, AutoTokenizer, OPTForCausalLM


# # Hebrew to english translator
# He_En_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
# He_En_tokenizer: MarianTokenizer = MarianTokenizer.from_pretrained(He_En_model_name)
# # He_En_translator_model = MarianMTModel.from_pretrained(He_En_model_name)


# # English to Hebrew translator
# En_He_model_name = "Helsinki-NLP/opus-mt-en-he"
# En_He_tokenizer: MarianTokenizer = MarianTokenizer.from_pretrained(En_He_model_name)
# # En_He_translator_model = MarianMTModel.from_pretrained(En_He_model_name)




def test(hebrew_dataset_path, stop_index=5):
    
    df = pd.read_csv(hebrew_dataset_path)
    
    for index, row in df.iterrows():
        if index > stop_index:
            break
        hebrew_sentence:str = row['Hebrew sentence']
        
        words = hebrew_sentence.split(' ')
        print(' '.join(words[:-1]))
        print(words[-1])

path = 'transformer_1/orel/sampled_data.csv'

test(path)
