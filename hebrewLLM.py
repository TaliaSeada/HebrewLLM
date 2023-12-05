"""Connect all files"""
from transformer_1 import activationsFromDifferentLayersTrans_1, trans2llamaNet
from transformer_2 import activationsFromDifferentLayersTrans_2, llama2transNet

# Input (from the user OR datasets IN HEBREW)
sen = input("Enter an Hebrew sentence: ")
# list_of_datasets = ["cities"] #[, "generated",  "inventions", "elements", "animals", "facts", "companies"]

# Hebrew to English translator (put the sentence through the translator)
layers_H2E = [-1]
translated_tokens, translated_sentence, embeddings = activationsFromDifferentLayersTrans_1.translate_sen(sen, layers_H2E[0])
print("\ntranslation: " + translated_sentence)

# Transformer 1 (convert the embeddings)
model1 = trans2llamaNet.reload(1024, 1024)
prediction1 = model1.predict(embeddings)
print(prediction1)

# llama2 (OPT)
# res =

# Transformer 2
# model2 = llama2transNet.reload(512, 512)
# prediction2 = model2.predict(embeddings)

# English to Hebrew translator
# layers_E2H = [1]
# translated_tokens2, translated_sentence2, embeddings2 = activationsFromDifferentLayersTrans_2.translate_sen(res, layers_E2H[0])

# Output
# print("\ntranslation: " + translated_sentence2)