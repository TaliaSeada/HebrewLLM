import torch
from torch import nn
from transformers import AutoTokenizer, OPTForCausalLM
import numpy as np


# TODO change this method - call the transformer
def your_input_modification(hidden_states):
    # loaded_array = np.load('tensor_data.npy')
    # tensor_data = torch.tensor(loaded_array)
    # modified_states = tensor_data

    modified_states = hidden_states
    # modified_states = hidden_states + 0.9999999999999
    # modified_states = hidden_states * 0
    return modified_states


# Load the tokenizer and model
model_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = OPTForCausalLM.from_pretrained(model_name)


class CustomLayerWrapper(nn.Module):
    def __init__(self, layer, hidden_states):
        super().__init__()
        self.layer = layer
        self.hs = hidden_states #transformer's result

    def forward(self, hidden_states, attention_mask=None, layer_head_mask=None,
                past_key_value=None, output_attentions=None, use_cache=None):
        # Apply modifications to hidden_states here
        modified_hidden_states = your_input_modification(self.hs)

        # Pass modified_hidden_states to the original layer
        return self.layer(modified_hidden_states, attention_mask, layer_head_mask,
                          past_key_value, output_attentions, use_cache)


def OPT_activation_different_layer(hidden_states, nlayer):
    # Create an instance of the layer you want to modify
    custom_layer = model.base_model.decoder.layers[nlayer]
    # Wrap the layer inside the custom wrapper
    wrapped_layer = CustomLayerWrapper(custom_layer, hidden_states)
    # Replace the layer with the wrapped layer
    model.base_model.decoder.layers[nlayer] = wrapped_layer

    # make dummy model
    # TODO depends on the input size
    inputs = tokenizer("It", return_tensors="pt")
    outputs = model(**inputs, output_hidden_states=True)

    # hs = outputs.hidden_states[1]
    # numpy_array = hs.detach().numpy()
    # np.save('tensor_data.npy', numpy_array)

    # Access the generated token IDs
    token_ids = outputs.logits.argmax(-1)
    # Decode the token IDs using the tokenizer
    generated_text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
    # Print the generated text
    print("Generated Text: ", generated_text)

    # Generate text
    # output = model.generate(**inputs, max_length=50, num_return_sequences=1)
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # print("Generated Text: ", generated_text)
    return outputs, generated_text

