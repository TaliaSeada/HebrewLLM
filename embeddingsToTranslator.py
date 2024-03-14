import torch
from torch import nn
from transformers import MarianMTModel, MarianTokenizer


# Define a function to modify hidden states
def your_input_modification(hidden_states):

    mod = hidden_states
    return mod


# Custom Layer Wrapper for MarianMT
class CustomLayerWrapper(nn.Module):
    def __init__(self, layer, hidden_states):
        super().__init__()
        self.layer = layer
        self.hs = hidden_states  # transformer's result

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # Modify hidden_states before passing them to the layer
        modified_hidden_states = your_input_modification(self.hs)

        # Ensure that attention_mask and any other necessary arguments are forwarded
        return self.layer(modified_hidden_states, attention_mask, **kwargs)


# Load the model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-he"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


def translator_activation_different_layer(hidden_states, nlayer=0):
    # Load the model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-en-he"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    original_layer = model.model.encoder.layers[nlayer]
    wrapped_layer = CustomLayerWrapper(original_layer, hidden_states)
    model.model.encoder.layers[nlayer] = wrapped_layer

    # TODO depends on the input size
    inputs = tokenizer("It", return_tensors="pt")
    decoder_start_token_id = tokenizer.pad_token_id
    decoder_input_ids = torch.full((inputs.input_ids.size(0), 1), decoder_start_token_id, dtype=torch.long).to(inputs.input_ids.device)

    # Custom model call
    outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_input_ids, output_hidden_states=True)

    generated_ids = model.generate(inputs.input_ids)

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # print("Generated Text: ", generated_text)
    return outputs, generated_text

# if __name__ == '__main__':
#     inputs = tokenizer("Dad", return_tensors="pt")
#     decoder_start_token_id = tokenizer.pad_token_id
#     decoder_input_ids = torch.full((inputs.input_ids.size(0), 1), decoder_start_token_id, dtype=torch.long).to(
#         inputs.input_ids.device)
#
#     outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
#
#     layer = 0
#     hs = outputs.encoder_hidden_states[layer]
#
#     output, generated_text = translator_activation_different_layer(hidden_states=hs, nlayer=layer)
#     print(generated_text)

