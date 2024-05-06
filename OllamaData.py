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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.metrics import edit_distance
from nltk.metrics import jaccard_distance


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


def calculate_cosine_similarity(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]


def calculate_edit_distance(text1, text2):
    return edit_distance(text1, text2)


def calculate_jaccard_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    return 1 - jaccard_distance(set1, set2)


if __name__ == '__main__':
    # Read data
    df = pd.read_csv('wikipedia_data.csv')

    # Set translators
    model_name_en2heb = "Helsinki-NLP/opus-mt-en-he"
    model_name_heb2en = "Helsinki-NLP/opus-mt-tc-big-he-en"
    device = "cpu"
    tokenizer_en2heb = AutoTokenizer.from_pretrained(model_name_en2heb)
    model_en2heb = AutoModelForSeq2SeqLM.from_pretrained(model_name_en2heb).to(device)

    tokenizer_heb2en = AutoTokenizer.from_pretrained(model_name_heb2en)
    model_heb2en = AutoModelForSeq2SeqLM.from_pretrained(model_name_heb2en).to(device)

    # set LLM
    # ollama.pull("phi3")
    model_to_use = "350m"
    model = OPTForCausalLM.from_pretrained("facebook/opt-" + model_to_use)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-" + model_to_use)

    similarity_scores = []
    for index, row in df.iterrows():
        if index == 1000:
            break
        user_input = row['Hebrew sentence']
        print(index)

        # Translate user input from Hebrew to English
        en_sent = translate_heb2en(user_input)

        # Get response in English
        # phi3
        # res = ollama.generate(model='llama2', prompt=en_sent)
        # en_response = res['response']

        # OPT-350m
        prompt = en_sent.rstrip(". ")
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(inputs.input_ids, max_length=512, num_return_sequences=1, num_beams=5,
                                 early_stopping=True)
        en_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Translate response from English to Hebrew
        heb_out = translate_en2heb(en_response)
        heb_out = heb_out.split(".")[0] + "."

        # Calculate similarity between the generated word and the real word
        real_word = row['label']
        cosine_sim = calculate_cosine_similarity(heb_out, real_word)
        edit_dist = calculate_edit_distance(heb_out, real_word)
        jaccard_sim = calculate_jaccard_similarity(heb_out, real_word)
        similarity_scores.append((cosine_sim, edit_dist, jaccard_sim))

        # Check the result
        print(user_input)
        print("Real word:", real_word)
        print("Generated sentence:", heb_out)
        # print("Cosine Similarity:", cosine_sim)
        # print("Edit Distance:", edit_dist)
        # print("Jaccard Similarity:", jaccard_sim)
        print("=" * 50)

    # Calculate average similarity score
    avg_cosine_sim = sum([score[0] for score in similarity_scores]) / len(similarity_scores)
    avg_edit_dist = sum([score[1] for score in similarity_scores]) / len(similarity_scores)
    avg_jaccard_sim = sum([score[2] for score in similarity_scores]) / len(similarity_scores)

    print("Average Cosine Similarity:", avg_cosine_sim)
    print("Average Edit Distance:", avg_edit_dist)
    print("Average Jaccard Similarity:", avg_jaccard_sim)
