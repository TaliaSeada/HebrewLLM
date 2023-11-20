import os
from transformers import AutoTokenizer, OPTForCausalLM
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = ''
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

layer_to_use = -5


# if the last token is a period, and there is no next token (not even sentence end), there will be one too many masks
def get_period_spaces_mask(input_text):
    period_indices = [i for i, char in enumerate(input_text) if char == '.' and (
            len(input_text) == i + 1 or input_text[i + 1] != '.')]  # ... should be counted as a single period
    mask = [len(input_text) == i + 1 or input_text[i + 1] == ' ' or input_text[i + 1] == '\n' for i in period_indices]
    return mask

#("facebook/opt-125m") #("facebook/opt-350m") #("facebook/opt-1.3b")
model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

# df = pd.read_csv("resources\\capitals.csv")
# res_df = pd.DataFrame()
prompt = None
for i in range(100):
    # sampled_row = df.sample(n=1)
    # city_name = sampled_row["city"].values[0]
    # prompt = city_name + " is a name of a city. " + city_name + " is located in"
    # Get prompt from user
    if prompt:
        user_prompt = prompt
        prompt = None
    else:
        user_prompt = input("Enter a prompt: ")

    if user_prompt.lower() == "n" or user_prompt.lower() == "stop" or user_prompt.lower() == "quit":
        break
    print("---Generating response...")
    # Tokenize the prompt
    inputs = tokenizer(user_prompt, return_tensors="pt")

    # Generate text
    outputs = model.generate(inputs.input_ids, return_dict_in_generate=True, max_new_tokens=150, min_new_tokens=20,
                             output_scores=True, no_repeat_ngram_size=3,
                             output_hidden_states=True)  # do_sample=True,, max_new_tokens=5, min_new_tokens=1) # return_logits=True, max_length=5, min_length=5, do_sample=True, temperature=0.5, no_repeat_ngram_size=3, top_p=0.92, top_k=10)return_logits=True
    generate_ids = outputs.sequences  # outputs[0]
    text = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    print(text)
