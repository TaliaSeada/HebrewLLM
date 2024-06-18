import pandas as pd
# # Read the CSV file
# df = pd.read_csv('wikipedia.csv', header=None)
#
# # Rename the current column to "Hebrew sentence"
# df.rename(columns={0: 'Hebrew sentence'}, inplace=True)
#
# # Extract the sixth word of each sentence and assign it to the label column
# df['label'] = df['Hebrew sentence'].str.split().str[5]
#
# # Extract the first five words of each sentence
# df['Hebrew sentence'] = df['Hebrew sentence'].str.split().str[:5].str.join(' ')
#
#
# # Drop the first row
# df = df.drop(0)
#
# # Save the modified DataFrame to a new CSV file
# df.to_csv('wikipedia_data.csv', index=False)

import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, OPTForCausalLM
import ollama


def translate_heb2en(text):
    tokenized_text = tokenizer_heb2en.encode(text, return_tensors='pt')
    translated_tokens = model_heb2en.generate(tokenized_text, max_length=512)
    return tokenizer_heb2en.decode(translated_tokens[0], skip_special_tokens=True)


def translate_en2heb(text):
    # Split the text into sentences
    sentences = nltk.sent_tokenize(text)

    translated_sentences = []
    for sentence in sentences:
        tokenized_sentence = tokenizer_en2heb.encode(sentence, return_tensors='pt')

        # Split the input into smaller chunks if it exceeds max_length
        max_length = 512
        if len(tokenized_sentence[0]) > max_length:
            input_chunks = [tokenized_sentence[0, i: i + max_length] for i in
                            range(0, len(tokenized_sentence[0]), max_length)]
        else:
            input_chunks = [tokenized_sentence[0]]

        translated_chunks = []
        for chunk in input_chunks:
            translated_tokens = model_en2heb.generate(chunk.unsqueeze(0), max_length=512, num_beams=5,
                                                      early_stopping=True)
            translated_chunk = tokenizer_en2heb.decode(translated_tokens[0], skip_special_tokens=True)
            translated_chunks.append(translated_chunk)

        translated_sentence = " ".join(translated_chunks)
        translated_sentences.append(translated_sentence)

    translated_text = " ".join(translated_sentences)
    return translated_text


if __name__ == '__main__':
    # Read data
    df = pd.read_csv('wikipedia_data.csv')

    # Set translators
    model_name_en2heb = "Helsinki-NLP/opus-mt-en-he"
    model_name_heb2en = "Helsinki-NLP/opus-mt-tc-big-he-en"

    tokenizer_en2heb = AutoTokenizer.from_pretrained(model_name_en2heb)
    model_en2heb = AutoModelForSeq2SeqLM.from_pretrained(model_name_en2heb)

    tokenizer_heb2en = AutoTokenizer.from_pretrained(model_name_heb2en)
    model_heb2en = AutoModelForSeq2SeqLM.from_pretrained(model_name_heb2en)

    # set LLM
    # ollama.pull("phi3")
    model_to_use = "350m"
    model = OPTForCausalLM.from_pretrained("facebook/opt-" + model_to_use)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-" + model_to_use)

    cnt = 0
    for index, row in df.iterrows():
        print(index)
        if index == 1000:
            break
        user_input = row['Hebrew sentence']

        # Translate user input from Hebrew to English
        en_sent = translate_heb2en(user_input)

        # Get response in English
        # phi3
        # res = ollama.generate(model='llama2', prompt=en_sent)
        # en_response = res['response']

        # OPT-350m
        prompt = en_sent.rstrip(". ")
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(inputs.input_ids,
                                 max_length=100,
                                 num_return_sequences=3,
                                 num_beams=5,
                                 top_k=50,
                                 early_stopping=True,
                                 do_sample=True,
                                 top_p=0.95,
                                 temperature=0.9)
        en_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = en_response[len(en_sent):].strip()

        # Translate response from English to Hebrew
        heb_out = translate_en2heb(result)
        if heb_out.strip():
            heb_out_word = heb_out.split()[0]
        else:
            heb_out_word = ""

        # Calculate similarity between the generated word and the real word
        real_word = row['label']

        # Check the result
        print(user_input)
        print("Real word:", real_word)
        print("Generated word:", heb_out_word)
        if real_word == heb_out_word:
            cnt += 1

        print("Counter: ", cnt)
        print("=" * 50)

    print("Success of ", (cnt / 1000) * 100, "%")
