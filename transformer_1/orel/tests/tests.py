
import joblib

from combinedTransformersModel import CombinedModel
from transformers import AutoTokenizer, AutoModel, MarianTokenizer, MarianMTModel, AutoTokenizer, OPTForCausalLM


# Hebrew to english translator
He_En_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
He_En_tokenizer: MarianTokenizer = MarianTokenizer.from_pretrained(He_En_model_name)
# He_En_translator_model = MarianMTModel.from_pretrained(He_En_model_name)


# English to Hebrew translator
En_He_model_name = "Helsinki-NLP/opus-mt-en-he"
En_He_tokenizer: MarianTokenizer = MarianTokenizer.from_pretrained(En_He_model_name)
# En_He_translator_model = MarianMTModel.from_pretrained(En_He_model_name)



def testCombined(h_text, model):
    
    logits = model(h_text)

    # print(logits)

    token_ids = logits.argmax(-1)

    generated_text = En_He_tokenizer.decode(token_ids[0], skip_special_tokens=True)

    return generated_text


def getNextWord(h_text, model, out_token_number = 1):
    
    input_token_size = (len(He_En_tokenizer(h_text,return_tensors="pt").input_ids[0])) - 1
    
    logits = model(h_text)

    # print(logits)

    token_ids = logits.argmax(-1)
    
    till_next_token = token_ids[0][:input_token_size + out_token_number]
    
    for index, token in enumerate(till_next_token):
        print(f"Token {index}: {En_He_tokenizer.decode(token, skip_special_tokens=True)}")
    
    generated_text = En_He_tokenizer.decode(till_next_token, skip_special_tokens=True)

    return generated_text
    
    
def printHeTokenizerIds(text: str):
    
    print(f"Hebrew text: {text}")
    
    # Translator
    inputs = He_En_tokenizer(text, return_tensors="pt")
    print(inputs.input_ids)


# combined_model = joblib.load(f'transformer_1/orel/pretrainedModels/models/combined/model_sampled_wiki_750_new_none_2words_learning.pkl')
combined_model = joblib.load(f'transformer_1/orel/pretrainedModels/models/15Tokens/model_wiki_30211_30210_new_none_2words_learning.pkl')
# combined_model = joblib.load(f'transformer_1/orel/pretrainedModels/models/15Tokens/model_wiki_30211_30210_new_none_5words_learning.pkl')



hebrew_words = ['של', 'את', 'על', 'הוא', 'ידי', 'היא', 'בין', 'עם', 'גם', 'כי', 'או',
       'היה', 'לא', 'כל', 'הייתה', 'בשנת', 'ביותר', 'יותר', 'עד', 'מספר',
       'היו', 'הם', 'יש', 'זו', 'זה', 'רק', 'באמצעות', 'כלל', 'אחד', 'שני',
       'ניתן', 'מאוד', 'באופן', 'העולם', 'רבים', 'לכל', 'בעיקר', 'הארץ', 'הן',
       'מערכת', 'אשר', 'אחת', 'חלק', 'שונים', 'רבות', 'מן', 'כאשר', 'כדי',
       'החל', 'השמש', 'כדור', 'בכל', 'במהלך', 'דרך', 'אותו', 'אלו', 'אל', 'כך',
       'כבר', 'פי', 'פני', 'תורת', 'שימוש', 'עבור', 'שלו', 'הברית', 'הראשונה',
       'בעלי', 'בשל', 'לפי', 'שלא', 'ישראל', 'בתוך', 'החיים', 'בשם', 'בארץ',
       'לפני', 'אך', 'רבה', 'בדרך', 'כוכבי']


# # for index, h_text in enumerate(hebrew_words):
    
    
# #     printHeTokenizerIds(h_text)
    
#     # generated_text = testCombined(h_text, combined_model)
    
#     # # Print the generated text
#     # print(f"Generated Output {index}: {generated_text}\n")
    
#     getNextWord(h_text, combined_model, 1)

# # print(En_He_tokenizer.get_vocab())