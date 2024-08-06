"""
This file is implementing the combination of the naive model and the second part of our mechanism (transformer OPT
to Translation)
We need the second transformer, the OPT model and the first translator
"""
import pandas as pd
import torch
from transformers import MarianTokenizer, MarianMTModel, OPTForCausalLM, AutoTokenizer

from transformer_2.model.multipleTokens.upto15Tokens import check_opt_hidden_state, HiddenStateTransformer
from transformer_2.model.multipleTokens.embeddToTrans15Tokens import translator_activation_different_layer


device = "cpu"
# opt
model_to_use = "350m"
opt_model = OPTForCausalLM.from_pretrained("facebook/opt-" + model_to_use)
opt_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-" + model_to_use, padding_side='left')
opt_layer = -1
# device = "cuda" if torch.cuda.is_available() else "cpu"
opt_model.to(device)

# translators
model_name_en2heb = "Helsinki-NLP/opus-mt-en-he"
model_name_heb2en = "Helsinki-NLP/opus-mt-tc-big-he-en"
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer_en2heb = MarianTokenizer.from_pretrained(model_name_en2heb)
model_en2heb = MarianMTModel.from_pretrained(model_name_en2heb).to(device)

tokenizer_heb2en = MarianTokenizer.from_pretrained(model_name_heb2en)
model_heb2en = MarianMTModel.from_pretrained(model_name_heb2en).to(device)
translator_layer = 1


def translate_heb2en(text):
    tokenized_text = tokenizer_heb2en.encode(text, return_tensors='pt').to(device)
    translated_tokens = model_heb2en.generate(tokenized_text, max_length=512)
    return tokenizer_heb2en.decode(translated_tokens[0], skip_special_tokens=True)


if __name__ == '__main__':
    '''-------------------------- input --------------------------'''
    data_path = 'C:\\Users\\talia\\PycharmProjects\\HebrewLLM\\translated_wikipedia_data.csv'
    df = pd.read_csv(data_path)
    df['Hebrew sentence'] = df['Hebrew sentence'].astype(str)

    for i, row in df.iterrows():
        prompt = row['Hebrew sentence']
        if i > 20:
            break

        '''-------------------------- translator --------------------------'''
        en_sent = translate_heb2en(prompt)
        print(en_sent)

        '''-------------------------- LLM --------------------------'''
        model = torch.load(
            'C:\\Users\\talia\\PycharmProjects\\HebrewLLM\\transformer_2\\model\\multipleTokens\\best_model.pth')
        num_of_tokens = 5
        generated_text, attention_mask, hidden_states = check_opt_hidden_state(en_sent, opt_model, opt_tokenizer,
                                                                               opt_layer)

        '''-------------------------- transformer --------------------------'''
        hidden_states = model(hidden_states)

        '''-------------------------- translator --------------------------'''
        # Call the function with hidden_states and attention_mask
        layer = 1
        outputs, res_generated_text = translator_activation_different_layer(hidden_states=hidden_states,
                                                                            attention_mask=attention_mask,
                                                                            nlayer=layer,
                                                                            max_length=num_of_tokens)

        '''-------------------------- output --------------------------'''
        print("Input: " + prompt)
        print("Output: " + res_generated_text)

        print("\n====================================================================================\n")
