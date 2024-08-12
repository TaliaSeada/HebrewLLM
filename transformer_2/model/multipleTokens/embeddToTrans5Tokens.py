import torch
from torch import nn
from transformers import MarianMTModel, MarianTokenizer


# Define a function to modify hidden states
def your_input_modification(hidden_states):
    # Ensure hidden_states has shape [batch_size, sequence_length, hidden_size]
    if hidden_states.dim() == 2:
        # Add batch dimension if missing
        hidden_states = hidden_states.unsqueeze(0)
    return hidden_states


# Custom Layer Wrapper for MarianMT
class CustomLayerWrapper(nn.Module):
    def __init__(self, layer, hidden_states):
        super().__init__()
        self.layer = layer
        self.hs = hidden_states  # transformer's result

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # Modify hidden_states before passing them to the layer
        modified_hidden_states = your_input_modification(self.hs)

        # Ensure hidden_states has the right shape
        if modified_hidden_states.dim() != 3:
            raise ValueError(f"Expected hidden_states to be 3D tensor, but got {modified_hidden_states.dim()}D tensor.")

        # Ensure that attention_mask and any other necessary arguments are forwarded
        return self.layer(modified_hidden_states, attention_mask, **kwargs)


# Load the model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-he"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


def translator_activation_different_layer(hidden_states, attention_mask, nlayer=1, max_length=5):
    original_layer = model.model.encoder.layers[nlayer]
    wrapped_layer = CustomLayerWrapper(original_layer, hidden_states)
    model.model.encoder.layers[nlayer] = wrapped_layer

    sentence = "0" * (max_length-1)
    inputs = tokenizer(sentence, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
    attention_mask = inputs['attention_mask']
    decoder_start_token_id = model.config.decoder_start_token_id
    decoder_input_ids = torch.full((inputs.input_ids.size(0), 1), decoder_start_token_id, dtype=torch.long).to(inputs.input_ids.device)

    outputs = model(input_ids=inputs.input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, output_hidden_states=True)

    eos_token_id = tokenizer.eos_token_id

    generated_ids = model.generate(
        inputs.input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=4,
        length_penalty=1.0,
        early_stopping=True,
        eos_token_id=eos_token_id
    )

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return outputs, generated_text


# if __name__ == '__main__':
#     # Test the function with multi-token input
#     sentence = "Maybe if you can"
#     max_length = 5
#     inputs = tokenizer(sentence, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
#     attention_mask = inputs['attention_mask']
#     decoder_start_token_id = model.config.decoder_start_token_id
#     decoder_input_ids = torch.full((inputs.input_ids.size(0), 1), decoder_start_token_id, dtype=torch.long).to(
#         inputs.input_ids.device)
#
#     outputs = model(input_ids=inputs.input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
#                     output_hidden_states=True)
#
#     layer = 1
#     hs = outputs.encoder_hidden_states[layer]
#     print(hs)
#
#     output, generated_text = translator_activation_different_layer(hidden_states=hs, attention_mask=attention_mask,
#                                                                    nlayer=layer, max_length=max_length)
