import torch
import torch.nn as nn
from transformers import AutoTokenizer, OPTForCausalLM, MarianMTModel, MarianTokenizer
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

# # translator english-hebrew
# model_name_en = "Helsinki-NLP/opus-mt-en-he"
# translator_model_en = AutoModelForSeq2SeqLM.from_pretrained(model_name_en)
# translator_tokenizer_en = AutoTokenizer.from_pretrained(model_name_en)
# translator_layer_en = 1

# translator hebrew-english
model_name_he = "Helsinki-NLP/opus-mt-tc-big-he-en"
translator_model_he = MarianMTModel.from_pretrained(model_name_he)
translator_tokenizer_he = MarianTokenizer.from_pretrained(model_name_he)
translator_layer_he = -1

if __name__ == '__main__':
    # Load data using the link above
    # TODO change to the link, TEMPORARY
    # data_path = 'C:\\Users\\talia\\PycharmProjects\\HebrewLLM\\resources\\dict.csv'
    # data_path = 'C:\\Users\\talia\\PycharmProjects\\HebrewLLM\\resources\\valid_guesses.csv'
    data_path = 'C:\\Users\\relwe\\OneDrive\\Documents\\GitHub\\HebrewLLM\\wiki_other_data.csv'

    df = pd.read_csv(data_path)
    df['label'] = df['label'].astype(str)
    # print(df)

    # Prepare data
    data = []
    N_goodWords = 0
    forAvg = [[],[],[]]
    for i, row in df.iterrows():
        if i<1000:
            continue
        if N_goodWords==2000:
            break
        prompt = row['label']
        
        # Translator last layer
        translator_inputs = translator_tokenizer_he(prompt, return_tensors="pt")
        translator_outputs = translator_model_he.generate(input_ids=translator_inputs.input_ids,
                                                              attention_mask=translator_inputs.attention_mask,
                                                              # decoder_input_ids=decoder_input_ids,
                                                              # max_length=16,  # adjust max_length as needed
                                                              num_beams=5,  # adjust num_beams as needed
                                                              early_stopping=True,
                                                              output_hidden_states=True)
        translated_text = translator_tokenizer_he.decode(translator_outputs[0], skip_special_tokens=True)
        
        # OPT first layer
        opt_inputs = opt_tokenizer(translated_text, return_tensors="pt")

        # Filter out long words
        if opt_inputs.input_ids.shape[1] != 2 or translator_outputs.shape[1] != 3:
            continue

        translator_inputs = translator_tokenizer_he.encode_plus(prompt, return_tensors='pt', truncation=True)
        
        
        # OPT
        opt_inputs = opt_tokenizer.encode_plus(translated_text,
                                                #    padding='max_length',
                                                #    max_length=N_maxTokens,  # adjust max_length as needed
                                                   return_tensors="pt")
        opt_outputs = opt_model(**opt_inputs, output_hidden_states=True)
        opt_hidden_state = opt_outputs.hidden_states[opt_layer][:,1,:].unsqueeze(1)
        
        # decoder_start_token_id = translator_tokenizer_he.pad_token_id
        # decoder_input_ids = torch.full((translator_inputs.input_ids.size(0), 1), decoder_start_token_id,
        #                                dtype=torch.long)
        translator_outputs = translator_model_he(input_ids=translator_inputs.input_ids,
                                              decoder_input_ids=translator_outputs, output_hidden_states=True)
        jhsgd = translator_outputs.decoder_hidden_states[translator_layer_he]
        for i in range(3):
            forAvg[i].append(translator_outputs.decoder_hidden_states[translator_layer_he][0,i,:])
        translator_hidden_state = translator_outputs.decoder_hidden_states[translator_layer_he].view(1,1,-1)
        
        # print(opt_hidden_state.shape, translator_hidden_state.shape)

        # # Filter out long words
        # if opt_hidden_state.shape[1] != 2 or translator_hidden_state.shape[1] != 2:
        #     continue
        
        # print(translated_text)

        N_goodWords += 1
        data.append((translator_hidden_state, opt_hidden_state))
        # print(data[-1])
    criterion = nn.MSELoss()
    for i in range(3):
        mean_values = torch.mean(forAvg[i], dim=0)
        # Expand the mean values back to the original shape
        avg_hs = mean_values.unsqueeze(0).expand_as(forAvg[i])
        print(criterion(avg_hs, forAvg[i]))
    torch.save(data, 'C:\\Users\\relwe\\OneDrive\\Documents\\GitHub\\HebrewLLM\\transformer_1\\hidden_state_transformer2500.pth')


    print("PTH file saved successfully as: trsltr_llm_1Token_Data")

    # # Convert data to DataFrame
    # df_final = pd.DataFrame(data, columns=['English', 'Hebrew'])

    # # Save DataFrame to CSV
    # csv_filename = 'English_one_token.csv'
    # df_final.to_csv(csv_filename, index=False)

    # print("CSV file saved successfully as:", csv_filename)