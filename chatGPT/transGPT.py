import os
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk

client = OpenAI(
    api_key="",
)


def call_llm(prompt, conversation):
    messages = [
        {
            "role": "user",
            "content": "You are ChatGPT a helpful assistant. You communicate in English only (i.e., your output "
                       "must be in English). Your input was translated from Hebrew, and your output will later "
                       "be translated back to Hebrew. Therefore, you must account for the following: (1) Use "
                       "only straightforward coreference, that is, each entity should be complete. (2) Avoid "
                       "Ambiguity. (3) Avoid Complex Sentence Structures. (4) Do not use Homonyms and "
                       "Homophones. (5) Avoid English-Specific Idioms and Slang. (6) Use Universal Examples and "
                       "References or those known to Jews, Israelis, and Hebrew speakers. ",
        }
    ]
    for (r, c) in conversation:
        messages.append({"role": r, "content": c})
    messages.append({"role": "user", "content": prompt})

    all_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        # temperature=0.9
    )

    response_st = all_response.choices[0].message.content
    conversation.append(("user", prompt))
    conversation.append(("assistant", response_st))

    return response_st



model_name_en2heb = "Helsinki-NLP/opus-mt-en-he"  # "Helsinki-NLP/opus-mt-tc-big-he-en" #"Helsinki-NLP/opus-mt-en-he"
model_name_heb2en = "Helsinki-NLP/opus-mt-tc-big-he-en"
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"
tokenizer_en2heb = AutoTokenizer.from_pretrained(model_name_en2heb)
model_en2heb = AutoModelForSeq2SeqLM.from_pretrained(model_name_en2heb).to(device)

tokenizer_heb2en = AutoTokenizer.from_pretrained(model_name_heb2en)
model_heb2en = AutoModelForSeq2SeqLM.from_pretrained(model_name_heb2en).to(device)


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


conversation = []
for i in range(10):
    sentence = input("משתמש/ת: ")
    if sentence.lower() == "quit" or sentence.lower() == "exit":
        break
    en_sent = translate_heb2en(sentence)
    print(en_sent)
    en_response = call_llm(en_sent, conversation)
    print(en_response)
    heb_out = translate_en2heb(en_response)
    print("מענה:" + heb_out)