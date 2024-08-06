import joblib
import torch
from combinedTransformersModel import CombinedModel
from transformers import AutoTokenizer, AutoModel, MarianTokenizer, MarianMTModel, AutoTokenizer, OPTForCausalLM
from generalTransformer import generate_test_inputs
from generalTransformer import CustomLayerWrapper, CustomLayerWrapper2



# # English to Hebrew translator
# En_He_model_name = "Helsinki-NLP/opus-mt-en-he"
# En_He_tokenizer = MarianTokenizer.from_pretrained(En_He_model_name)


# model = joblib.load('transformer_1\orel\pretrainedModels\models\combined\model_sampled_wiki_1.pkl')


# hebrew_sentence = "אני ממש רוצה ללכת"
# target_hebrew_sentence = hebrew_sentence + " " + "לישון"

# with torch.no_grad():  # No gradient calculation
#     # Outputs predicted distribution for each token
#     q = model(hebrew_sentence)


#     top_n_logits = torch.topk(q, 3, dim=-1).indices

#     # Decode the top 3 logits into words
#     top_n_words = []
#     for i in range(top_n_logits.size(1)):
#         words = [En_He_tokenizer.decode([token_id.item()]) for token_id in top_n_logits[0, i]]
#         top_n_words.append(words)
#     print(f"Output sentence = {top_n_words}")


# LLM model
llm_model_name = "facebook/opt-350m"
llm_tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm:OPTForCausalLM = OPTForCausalLM.from_pretrained(llm_model_name)


h_text = "למרבית הדפדפנים יש יכולת להתקשר"
h_text = "אני רוצה ללכת"
h_text = "ניתן"
h_text = "אפשר"
h_text = "כן"
h_text = "לא"
h_text = "ה"

trans_last_hs_padded, opt_first_hs_padded, input_tokens_len = generate_test_inputs(h_text)

t1 = joblib.load('transformer_1/orel/pretrainedModels/models/15Tokens\model_wiki_10414_36000.pkl')


t1.eval()
with torch.no_grad():
    hidden_states = t1(trans_last_hs_padded)
    
    # Trick llm
    inputs = llm_tokenizer(" " * 14, return_tensors="pt")
    original_layer = llm.base_model.decoder.layers[1]
    wrapped_layer = CustomLayerWrapper(original_layer, None)
    llm.base_model.decoder.layers[1] = wrapped_layer
    
    # Inject our initial embeddings to the first layer of the llm
    llm.base_model.decoder.layers[1].hs = hidden_states
                
    # Calculates the final embeddings
    outputs = llm(**inputs, output_hidden_states=True)
    
    # print(outputs.keys())
    
    # # Get the top logits for each position
    # top_n_logits = torch.topk(outputs.logits, 1, dim=-1).indices

    # # print(top_n_logits)
    
    # print("\n====== OPT generated text after using transformer 1: ======\n")

    # # Decode the top 3 logits into words
    # top_n_words = []
    # for i in range(top_n_logits.size(1)):
    #     for token_id in top_n_logits[0, i]:
    #         print(f"{top_n_logits[0, i]} = {[llm_tokenizer.decode([token_id.item()])]}\n") 
        # top_n_words.append(words)

    # llm_tokenizer.as_target_tokenizer():

    # print(top_n_words)
    
    # Access the generated token IDs
    token_ids = outputs.logits.argmax(-1)

    generated_text = llm_tokenizer.decode(token_ids[0], skip_special_tokens=True)
    
    # Print the generated text
    print("Generated Text: ", generated_text)