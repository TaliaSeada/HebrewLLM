import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import ollama


def translate_heb2en(text):
    tokenized_text = tokenizer_heb2en.encode(text, return_tensors='pt').to(device)
    translated_tokens = model_heb2en.generate(tokenized_text, max_length=512)
    return tokenizer_heb2en.decode(translated_tokens[0], skip_special_tokens=True)


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


if __name__ == '__main__':
    # Set translators
    model_name_en2heb = "Helsinki-NLP/opus-mt-en-he"
    model_name_heb2en = "Helsinki-NLP/opus-mt-tc-big-he-en"
    device = "cpu"
    tokenizer_en2heb = AutoTokenizer.from_pretrained(model_name_en2heb)
    model_en2heb = AutoModelForSeq2SeqLM.from_pretrained(model_name_en2heb).to(device)

    tokenizer_heb2en = AutoTokenizer.from_pretrained(model_name_heb2en)
    model_heb2en = AutoModelForSeq2SeqLM.from_pretrained(model_name_heb2en).to(device)

    # ollama.pull("phi3")
    conversation = []
    while True:
        user_input = input("משתמש/ת: ")
        if user_input.lower() == "quit" or user_input.lower() == "exit":
            break

        # Translate user input from Hebrew to English
        en_sent = translate_heb2en(user_input)
        if en_sent.lower() == "quit" or en_sent.lower() == "exit":
            break
        print("English: " + en_sent)
        conversation.append(("משתמש/ת:", user_input))

        # Concatenate conversation history with user input
        conversation_history = '\n'.join([f'{role}: {utterance}' for role, utterance in conversation])
        prompt = en_sent + "\n" + conversation_history

        # Get response in English
        # ollama.pull("llama2")
        res = ollama.generate(model='llama2', prompt=prompt)
        en_response = res['response']
        print("English Response: " + en_response)
        conversation.append(("מערכת:", en_response))

        # Translate response from English to Hebrew
        heb_out = translate_en2heb(en_response)
        print("מענה:" + heb_out)
        conversation.append(("מערכת (Hebrew):", heb_out))

    print("השיחה הסתיימה. תודה רבה!")

    # Print full conversation
    print("\nהשיחה המלאה:")
    for role, utterance in conversation:
        print(role, utterance)


