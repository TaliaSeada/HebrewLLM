from transformers import AutoTokenizer, OPTForCausalLM, AutoConfig
import pandas as pd
import torch

import hebrewLLM


class CustomOPTForCausalLM(OPTForCausalLM):
    def __init__(self, config):
        super().__init__(config)


    def forward(self, input_ids=None, custom_embeddings=None, **kwargs):
        if custom_embeddings is not None:
            # Use the forward method of the base class with inputs_embeds
            return super().forward(inputs_embeds=custom_embeddings, **kwargs)
        else:
            # Use the original forward method if no custom embeddings are provided
            return super().forward(input_ids=input_ids, **kwargs)


# model_to_use = "350m"
# model = OPTForCausalLM.from_pretrained(f"facebook/opt-{model_to_use}")
# tokenizer = AutoTokenizer.from_pretrained(f"facebook/opt-{model_to_use}")
#
# # embeddings = torch.randn(1, 50, 768)
#
# # Dummy input to go through the initial steps of the model
# dummy_input = tokenizer("It", return_tensors="pt").input_ids
#
# # Forward pass through the model, up to the embedding layer
# outputs = model(dummy_input, output_hidden_states=True)
# hidden_states = outputs.hidden_states
#
# print(hidden_states[0])
# Replace the first hidden state (embeddings) with your custom embeddings
# hidden_states[0] = embeddings

#
# ("facebook/opt-125m") #("facebook/opt-350m") #("facebook/opt-1.3b")
config = AutoConfig.from_pretrained(f"facebook/opt-350m")
custom_model = CustomOPTForCausalLM(config=config)


# load data
file = "wordsTransEmbedd.csv"
data = pd.read_csv(file)

# OPT
words = ["שלום", "זה", "לי", "למה", "איפה", "אפילו"]

for word in words:
    custom_embeddings = hebrewLLM.pred(word)

    # Convert to PyTorch tensor
    custom_embeddings_tensor = torch.tensor(custom_embeddings, dtype=torch.float)

    # Debugging: Print the hidden size of the model
    hidden_size = config.hidden_size
    print("Model's hidden size:", hidden_size)

    # Calculate sequence_length
    sequence_length = custom_embeddings_tensor.shape[1] // hidden_size
    if custom_embeddings_tensor.shape[1] % hidden_size != 0:
        print(
            f"Warning: The total number of features in the embeddings ({custom_embeddings_tensor.shape[1]}) is not a multiple of the model's hidden size ({hidden_size}).")

    # Reshape embeddings
    custom_embeddings_tensor = custom_embeddings_tensor.view(1, sequence_length, hidden_size)

    # Debugging: Print type and shape
    print("Type of custom_embeddings:", type(custom_embeddings_tensor))
    print("Shape of custom_embeddings:", custom_embeddings_tensor.shape)

    # Forward pass
    output = custom_model(custom_embeddings=custom_embeddings_tensor)
    print(output)


