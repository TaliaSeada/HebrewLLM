

import joblib
import pandas as pd

from transformers import AutoTokenizer, AutoModel, MarianTokenizer, MarianMTModel, AutoTokenizer, OPTForCausalLM


# Hebrew to english translator
He_En_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
He_En_tokenizer: MarianTokenizer = MarianTokenizer.from_pretrained(He_En_model_name)
He_En_translator_model = MarianMTModel.from_pretrained(He_En_model_name)

# LLM model
llm_model_name = "facebook/opt-350m"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm:OPTForCausalLM = OPTForCausalLM.from_pretrained(llm_model_name)

# English to Hebrew translator
En_He_model_name = "Helsinki-NLP/opus-mt-en-he"
En_He_tokenizer: MarianTokenizer = MarianTokenizer.from_pretrained(En_He_model_name)
En_He_translator_model = MarianMTModel.from_pretrained(En_He_model_name)


"""
    This is the direct approach:
    [Hebrew Sentence -> OPT -> Hebrew Output]
"""
def optNextWord(hebrew_input):
    
    llm_inputs = llm_tokenizer(hebrew_input, return_tensors="pt")
    llm_out_ids = llm.generate(llm_inputs.input_ids)
    llm_answer = llm_tokenizer.decode(llm_out_ids[0],skip_special_tokens=True)
    # print(f"LLM answer: {llm_answer}")

    input_size = len(hebrew_input.split(' '))
    
    answer_words = llm_answer.split(' ')
    
    return answer_words[input_size] if len(answer_words) > input_size else None


"""
    This is the basic 2 way translation (Not trainable):
    [Hebrew Sentence -> Translator -> OPT -> Translator -> Hebrew Output] 
"""
def basicModelNextWord(hebrew_input):
    
    # print(f"Hebrew input: {hebrew_input}")
    
    # Translate to english
    inputs = He_En_tokenizer(hebrew_input, return_tensors="pt")
    en_ids = He_En_translator_model.generate(inputs.input_ids)
    en_trans = He_En_tokenizer.decode(en_ids[0], skip_special_tokens=True)
    # print(f"English translation: {en_trans}")
    
    llm_inputs = llm_tokenizer(en_trans, return_tensors="pt")
    llm_out_ids = llm.generate(llm_inputs.input_ids)
    llm_answer = llm_tokenizer.decode(llm_out_ids[0],skip_special_tokens=True)
    # print(f"LLM answer: {llm_answer}")
    
    # Translate back to hebrew
    en_inputs = En_He_tokenizer(llm_answer, return_tensors="pt")
    he_ids = En_He_translator_model.generate(en_inputs.input_ids)
    he_trans = En_He_tokenizer.decode(he_ids[0], skip_special_tokens=True)
    # print(f"Hebrew translation: {he_trans}")

    input_size = len(hebrew_input.split(' '))
    
    answer_words = he_trans.split(' ')
    
    return answer_words[input_size] if len(answer_words) > input_size else None


"""
    This is the full model:
    [Hebrew Sentence -> Translator -> Transformer 1 -> OPT -> Transformer 2 -> Translator -> Hebrew Output]
"""
def fullModelNextWord(h_text, model):

    # Call pretrained combined model with hebrew input
    model.eval()
    logits = model(h_text)

    # Extract sentence - Translate back to hebrew
    token_ids = logits.argmax(-1)
    output_sentence = En_He_tokenizer.decode(token_ids[0], skip_special_tokens=True)

    # Return next word
    input_size = len(h_text.split(' '))
    output_words = output_sentence.split(' ')
    
    return output_words[input_size] if len(output_words) > input_size else None


def test(hebrew_dataset_path, model_type = "basic", model = None, stop_index=float('inf'), input_size = 1):
    
    if input_size < 1:
        print("Input size must be > 1")
        return
    
    df = pd.read_csv(hebrew_dataset_path)

    success_counter = 0
    test_size = 0
    
    # Iterate dataset rows
    for index, row in df.iterrows():
        if index >= stop_index:
            break
        
        hebrew_sentence:str = row['Hebrew sentence']
        
        words = hebrew_sentence.split(' ')
        
        # Ensure we are within scope
        input_size = min(input_size, len(words))
        
        he_input = " ".join(words[:input_size])
        target = words[input_size]
        
        actual = None
        
        if model_type == "basic":
            # Basic 2 way translation model
            actual = basicModelNextWord(he_input)
        elif model_type == "full":
            # Our pretrained model 
            actual = fullModelNextWord(model, he_input)
        else:
            # Direct hebrew to OPT
            actual = optNextWord(he_input)

        if actual:
            if actual == target:
                success_counter += 1
            test_size += 1

        if index % 1000 == 0:
            print(f"Current test size = {test_size}/{index + 1}, Success: {success_counter}")
    print(f"Test size = {test_size}/{min(stop_index, df.shape[0])}, Success: {success_counter}")
