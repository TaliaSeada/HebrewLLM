import pandas as pd
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, OPTForCausalLM
import ollama
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.metrics import edit_distance
from nltk.metrics import jaccard_distance


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

    # set LLM
    # ollama.pull("phi3")
    model_to_use = "350m"
    model = OPTForCausalLM.from_pretrained("facebook/opt-" + model_to_use)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-" + model_to_use)

    similarity_scores = []
    generated_responses = set()
    for index, row in df.iterrows():
        print(index)
        if index == 101:
            break
        user_input = row['Hebrew sentence']

        # Get response in Hebrew
        inputs = tokenizer(user_input, return_tensors="pt", max_length=1024, truncation=True, padding=True)
        outputs = model.generate(inputs.input_ids,
                                 max_length=100,
                                 num_return_sequences=3,
                                 num_beams=5,
                                 top_k=50,
                                 early_stopping=True,
                                 do_sample=True,  # Enable sampling
                                 top_p=0.95,
                                 temperature=0.9)

        # Filter out repetitive and similar sentences
        responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs if
                     tokenizer.decode(output, skip_special_tokens=True) != user_input and tokenizer.decode(
                         output, skip_special_tokens=True) not in generated_responses]
        generated_responses.update(responses)
        response = responses[0] if responses else tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Calculate similarity between the generated word and the real word
        real_word = row['label']
        cosine_sim = calculate_cosine_similarity(response, real_word)
        edit_dist = calculate_edit_distance(response, real_word)
        jaccard_sim = calculate_jaccard_similarity(response, real_word)
        similarity_scores.append((cosine_sim, edit_dist, jaccard_sim))

        # Check the result
        print("User input:", user_input)
        print("Real word:", real_word)
        print("Generated sentence:", response)
        print("=" * 50)

    # Calculate average similarity score
    avg_cosine_sim = sum([score[0] for score in similarity_scores]) / len(similarity_scores)
    avg_edit_dist = sum([score[1] for score in similarity_scores]) / len(similarity_scores)
    avg_jaccard_sim = sum([score[2] for score in similarity_scores]) / len(similarity_scores)

    print("Average Cosine Similarity:", avg_cosine_sim)
    print("Average Edit Distance:", avg_edit_dist)
    print("Average Jaccard Similarity:", avg_jaccard_sim)
