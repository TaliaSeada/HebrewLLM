# import random
# import requests

# # Example Hebrew sentence and target word
# # sentence = ["אני", "אוהב", "ללמוד", "בינה", "מלאכותית"]
# # target_word = "שפה"

# sentence = ["I", "like", "to"]
# target_word = "do"

# # Function to get synonyms using dictionaryapi.dev
# def get_hebrew_synonyms(word):
#     url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
#     response = requests.get(url)
#     synonyms = set()
#     if response.status_code == 200:
#         data = response.json()
#         # print(data)
#         if 'meanings' in data[0]:
#             for meaning in data[0]['meanings']:
#                 if 'synonyms' in meaning:
#                     for synonym in meaning['synonyms']:
#                         synonyms.add(synonym)
#     else:
#         print(response.status_code)
#     return list(synonyms)

# def synonym_replacement(words, n):
#     new_words = words.copy()
#     random_word_list = list(set([word for word in words if get_hebrew_synonyms(word)]))
#     random.shuffle(random_word_list)
#     num_replaced = 0
#     for random_word in random_word_list:
#         synonyms = get_hebrew_synonyms(random_word)
#         if len(synonyms) >= 1:
#             synonym = random.choice(synonyms)
#             new_words = [synonym if word == random_word else word for word in new_words]
#             num_replaced += 1
#         if num_replaced >= n:
#             break

#     return new_words

# # Perform data augmentation
# augmented_sentence_synonym = synonym_replacement(sentence, 1)

# print("Original sentence:", sentence)
# print("Target word:", target_word)
# print("Augmented sentence with synonym replacement:", augmented_sentence_synonym)


import nlpaug.augmenter.word as naw

# Example Hebrew sentence
sentence = "אני אוהב ללמוד בינה מלאכותית"

# Initialize the synonym augmenter
aug = naw.SynonymAug(aug_src='wordnet')

# Perform synonym replacement
augmented_sentence = aug.augment(sentence)

print("Original sentence:", sentence)
print("Augmented sentence with synonym replacement:", augmented_sentence)

