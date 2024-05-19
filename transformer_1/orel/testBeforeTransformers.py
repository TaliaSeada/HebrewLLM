import pandas as pd
import numpy as np
import torch
from transformers import MarianTokenizer,MarianMTModel, AutoTokenizer, OPTForCausalLM,AutoModel
import csv
import time
import os
import h5py


device = "cuda" if torch.cuda.is_available() else "cpu"


# Hebrew to english translator
translator_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"

translator_tokenizer = MarianTokenizer.from_pretrained(translator_model_name)
translator_model = MarianMTModel.from_pretrained(translator_model_name)

# OPT model
OPT_model_name = "facebook/opt-350m"
OPT_tokenizer = AutoTokenizer.from_pretrained(OPT_model_name)
opt_model = AutoModel.from_pretrained(OPT_model_name).to(device)





hebrew_text = "ואיך שלא אפנה לראות תמיד איתה ארצה להיות שומרת לי היא אמונים לא מתרוצצת בגנים וגם אני בין הבריות לא מתפתה לאחרות גלים עולים חולות נעים אוושת הרוח בעלים ובלילות ובלילות עולות עולות בי מנגינות וזרם דק קולח ותפילותי לרוח נענות שקט עכשיו וכל אחד עסוק בענייניו אל הרופא הכי טוב במדינה בכדי לחשוף עד הסוף אך התמונה לא משתנה מילים מילים ואת משמעותן יבוא לו גל לשטוף אותן אבל אני כרוך אחריה היא מחכה לי בלילות ללכת שבי אחריה לשמוע את הציפורים שרות גלים עולים חולות נעים אוושת הרוח בעלים ואיך שלא אפנה לראות תמיד איתה ארצה להיות"



# Translator
inputs = translator_tokenizer(hebrew_text, return_tensors="pt")

# Encode the source text
generated_ids = translator_model.generate(inputs.input_ids)

# Translation to english
english_text = translator_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# print(english_text)

english_words = english_text.split(' ')

print(english_words)
text = english_words[0]

for i, word in enumerate(english_words):
    if i > 0:
        text += " " + word
    
    
    inputs = OPT_tokenizer("It", return_tensors="pt")
    outputs = opt_model(**inputs, output_hidden_states=True)

    # hs = outputs.hidden_states[1]
    # numpy_array = hs.detach().numpy()
    # np.save('tensor_data.npy', numpy_array)

    # Access the generated token IDs
    token_ids = outputs.logits.argmax(-1)
    # Decode the token IDs using the tokenizer
    generated_text = OPT_tokenizer.decode(token_ids[0], skip_special_tokens=True)
    # Print the generated text
    print("Generated Text: ", generated_text)