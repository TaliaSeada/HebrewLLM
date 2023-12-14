from transformers import AutoModelForSeq2SeqLM
import nltk
import os
from transformers import AutoTokenizer, OPTForCausalLM

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# "Helsinki-NLP/opus-mt-tc-big-he-en" #"Helsinki-NLP/opus-mt-en-he"
model_name_en2heb = "Helsinki-NLP/opus-mt-en-he"
model_name_heb2en = "Helsinki-NLP/opus-mt-tc-big-he-en"
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"

tokenizer_en2heb = AutoTokenizer.from_pretrained(model_name_en2heb)
model_en2heb = AutoModelForSeq2SeqLM.from_pretrained(model_name_en2heb).to(device)

tokenizer_heb2en = AutoTokenizer.from_pretrained(model_name_heb2en)
model_heb2en = AutoModelForSeq2SeqLM.from_pretrained(model_name_heb2en).to(device)

# ("facebook/opt-125m") #("facebook/opt-350m") #("facebook/opt-1.3b")
model_engine = OPTForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")


# if the last token is a period, and there is no next token (not even sentence end), there will be one too many masks
def get_period_spaces_mask(input_text):
    period_indices = [i for i, char in enumerate(input_text) if char == '.' and (
            len(input_text) == i + 1 or input_text[i + 1] != '.')]  # ... should be counted as a single period
    mask = [len(input_text) == i + 1 or input_text[i + 1] == ' ' or input_text[i + 1] == '\n' for i in period_indices]
    return mask


def call_ai(prompt):
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate text
    outputs = model_engine.generate(inputs.input_ids, return_dict_in_generate=True, max_new_tokens=150,
                                    min_new_tokens=20, output_scores=True, no_repeat_ngram_size=3,
                                    output_hidden_states=True)
    generate_ids = outputs.sequences  # outputs[0]
    text = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

    return text


def call_ai_embeddings(embeddings):
    # make sentence out of the embeddings
    sen = ""

    return call_ai(sen)


def translate_en2heb(text):
    # Split the text into sentences
    sentences = nltk.sent_tokenize(text)

    translated_sentences = []
    for sentence in sentences:
        tokenized_sentence = tokenizer_en2heb.encode(sentence, return_tensors='pt').to(device)
        translated_tokens = model_en2heb.generate(tokenized_sentence, max_length=512)
        translated_sentence = tokenizer_en2heb.decode(translated_tokens[0], skip_special_tokens=True)
        translated_sentences.append(translated_sentence)

    translated_text = " ".join(translated_sentences)
    return translated_text


def translate_heb2en(text):
    tokenized_text = tokenizer_heb2en.encode(text, return_tensors='pt').to(device)
    translated_tokens = model_heb2en.generate(tokenized_text, max_length=512)
    return tokenizer_heb2en.decode(translated_tokens[0], skip_special_tokens=True)

# for i in range(10):
#     sentence = input("משתמש/ת: ")
#     if sentence.lower() == "quit" or sentence.lower() == "exit" or sentence == "יציאה":
#         break
#     en_sent = translate_heb2en(sentence)
#     print(en_sent)
#     en_response = call_ai(en_sent)
#     print(en_response)
#     heb_out = translate_en2heb(en_response)
#     print("מענה:" + heb_out)
