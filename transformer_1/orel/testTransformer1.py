import torch
import joblib
from transformers import AutoTokenizer, AutoModel, MarianTokenizer,MarianMTModel, AutoTokenizer, OPTForCausalLM
from generalTransformer import CustomLayerWrapper
from combinedTransformersModel import hebrew_to_input
import pandas as pd



def inject_layer(model, layer_hs, layer_num, name="decoder"):
    if name == "decoder":
        original_layer = model.base_model.decoder.layers[layer_num]
    else:
        original_layer = model.model.encoder.layers[layer_num]
    
    # Wrap the layer inside the custom wrapper
    wrapped_layer = CustomLayerWrapper(original_layer, layer_hs)
    
    # Replace the layer with the wrapped layer
    if name == "decoder":
        model.base_model.decoder.layers[layer_num] = wrapped_layer
    else:
        model.model.encoder.layers[layer_num] = wrapped_layer
            
            


def get_next_word(text, t1, He_En_tokenizer, He_En_translator_model, llm, En_He_tokenizer, En_He_translator_model):
    # print(f"Input text = {text}")

    with torch.no_grad():
        # Get the final emmbeding of translator1 for the text input (language1)  
        x, clean_token_num = hebrew_to_input(text, He_En_tokenizer, He_En_translator_model)

        # Transform it to the llm initial embeddings for the sentence in language2 
        x = t1(x)
        # print(x)
        # print(x.shape)

            
        # Trick llm by giving it a dummy that contain the desired number of tokens (In out case 15)
        # and replace the first layer as it got other word embedding.
        inputs = llm_tokenizer(" " * 14, return_tensors="pt")
        
        # print(f"inputs = {inputs}")
        
        # ================ Inject ==============
        # Inject our initial embeddings to the first layer of the llm    
        original_layer = llm.base_model.decoder.layers[1]

        # Wrap the layer inside the custom wrapper
        wrapped_layer = CustomLayerWrapper(original_layer, x)
        
        # Replace the layer with the wrapped layer
        llm.base_model.decoder.layers[1] = wrapped_layer
        
        # ================ Inject ==============
        
        outputs = llm(**inputs, output_hidden_states=True)
        
        # ================ LLM ===============
        # Access the generated token IDs
        token_ids = outputs.logits.argmax(-1)
        # Decode the token IDs using the tokenizer
        translated_sentence = llm_tokenizer.decode(token_ids[0], skip_special_tokens=True)
        

        print(f"LLM output: {translated_sentence}")
        # ================ LLM ===============

        
        # ================ Translator ================
        en_inputs = En_He_tokenizer(translated_sentence, return_tensors="pt")
        print(f"en_inputs = {en_inputs}")
        
        # new_token = En_He_tokenizer.convert_ids_to_tokens(en_inputs.input_ids[0])[clean_token_num + 1]
        new_token_id = en_inputs.input_ids[0,clean_token_num + 1]
        
        # Encode the source text
        outputs2: torch.Tensor = En_He_translator_model.generate(en_inputs.input_ids)
        
        print(f"Output ids = {outputs2[0]}")
        translated_to_hebrew = En_He_tokenizer.decode(outputs2[0], skip_special_tokens=True)
        
        for t in outputs2:
            print(f"translated token {t} = {En_He_tokenizer.convert_ids_to_tokens(t)}")

        
        print(f"Final output sentence = {translated_to_hebrew}")
        
        # Decode the single token ID
        translated_new_word = En_He_tokenizer.decode([new_token_id], skip_special_tokens=True)

        print(f"New word = {translated_new_word}")
        # ================ Translator ================

        return translated_new_word


def test_transformer(dataset_path, t1, He_En_tokenizer, He_En_translator_model, llm, En_He_tokenizer, En_He_translator_model):
    
    df = pd.read_csv(dataset_path)
    df = df.dropna()
    
    for index, row in df.iterrows():
        if index > 0: break
    
        he_sentence = row[0]
        target_word = row[1]
        actual_word = get_next_word(he_sentence,t1, He_En_tokenizer, He_En_translator_model, llm, En_He_tokenizer, En_He_translator_model)
        print(f"Actual: {actual_word}, target: {target_word}")
    # text = ""
    # next_word = get_next_word(text, t1, He_En_tokenizer, He_En_translator_model, llm, En_He_tokenizer, En_He_translator_model)


text = "אני הולך"



# Hebrew to english translator
He_En_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"

He_En_tokenizer = MarianTokenizer.from_pretrained(He_En_model_name)
He_En_translator_model = MarianMTModel.from_pretrained(He_En_model_name)



# Transformer 1
t1 = joblib.load('transformer_1/orel/pretrainedModels/models/10Tokens/general_model.pkl')


# LLM model
llm_model_name = "facebook/opt-350m"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm = OPTForCausalLM.from_pretrained(llm_model_name)


# English to Hebrew translator
En_He_model_name = "Helsinki-NLP/opus-mt-en-he"
En_He_tokenizer: MarianTokenizer = MarianTokenizer.from_pretrained(En_He_model_name)
En_He_translator_model: MarianMTModel = MarianMTModel.from_pretrained(En_He_model_name)


dataset_path = 'wikipedia_data.csv'
dataset_path = 'C:\\Users\\orelz\\OneDrive\\\שולחן העבודה\\work\\Ariel\\HebrewLLM\\wikipedia_data.csv'

# test_transformer(dataset_path=dataset_path,
#                 t1=t1,
#                 He_En_tokenizer=He_En_tokenizer,
#                 He_En_translator_model=He_En_translator_model,
#                 llm=llm,
#                 En_He_tokenizer=En_He_tokenizer,
#                 En_He_translator_model=En_He_translator_model)


text = "אתה הולך"

get_next_word(text=text,
                t1=t1,
                He_En_tokenizer=He_En_tokenizer,
                He_En_translator_model=He_En_translator_model,
                llm=llm,
                En_He_tokenizer=En_He_tokenizer,
                En_He_translator_model=En_He_translator_model)