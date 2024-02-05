# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#
# model_name = "Helsinki-NLP/opus-mt-en-he"
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# input_ids = tokenizer.encode("Hello", return_tensors="pt")
# outputs = model.generate(input_ids, output_hidden_states=True, return_dict_in_generate=True, max_length=512)
# print("Generated:", tokenizer.decode(outputs[0][0], skip_special_tokens=True))
#
# # The last element of the tuple contains the hidden states
# layer = 0
# hidden_states = outputs.encoder_hidden_states[layer]
#
# # Use the hidden states as input for another forward pass
# new_input_ids = torch.argmax(hidden_states, dim=-1)  # Convert hidden states to token indices
# new_outputs = model.generate(new_input_ids, max_length=512)
#
# # Decode and print the generated text
# generated_text = tokenizer.decode(new_outputs[0][0], skip_special_tokens=True)
# print("Generated:", generated_text)
#------------------------------------------------------------------------------------------------------
# from transformers import MarianMTModel, MarianTokenizer, MarianConfig
# import torch
# from torch import nn
#
# def your_input_modification(hidden_states):
#     modified_states = hidden_states
#     return modified_states
#
# class CustomMarianMTModel(MarianMTModel):
#     def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, encoder_outputs=None,
#                 past_key_values=None, inputs_embeds=None, decoder_inputs_embeds=None, **kwargs):
#
#         modified_hidden_states = your_input_modification(inputs_embeds)
#
#         return super().forward(input_ids=input_ids, attention_mask=attention_mask,
#                                decoder_input_ids=decoder_input_ids, encoder_outputs=encoder_outputs,
#                                past_key_values=past_key_values, inputs_embeds=inputs_embeds,
#                                decoder_inputs_embeds=decoder_inputs_embeds, **kwargs)
#
# # Load the tokenizer and model
# model_name = "Helsinki-NLP/opus-mt-en-he"
# tokenizer = MarianTokenizer.from_pretrained(model_name)
# config = MarianConfig.from_pretrained(model_name)
# custom_model = CustomMarianMTModel.from_pretrained(model_name, config=config)
# model = MarianMTModel.from_pretrained(model_name)
#
# # Encode the input text
# input_text = "Talia"
# input_ids = tokenizer.encode(input_text, return_tensors="pt")
#
# # Generate initial outputs
# outputs = model.generate(input_ids=input_ids, output_hidden_states=True, max_length=512, return_dict_in_generate=True)
#
# generated_text = tokenizer.decode(outputs[0][0], skip_special_tokens=True)
# print("Generated:", generated_text)
#
# # Access the hidden states from the encoder
# layer = 0
# hidden_states = outputs.encoder_hidden_states[layer]
#
# # Use the hidden states as input for another forward pass
# new_outputs = custom_model.generate(inputs_embeds=hidden_states, output_hidden_states=True, max_length=512, return_dict_in_generate=True)
#
# # Decode and print the generated text
# generated_text = tokenizer.decode(new_outputs[0][0], skip_special_tokens=True)
# print("Custom Generated:", generated_text)
#-----------------------------------------------------------------------------------------------------------
import inspect

import torch
from torch import nn
from transformers import MarianMTModel, MarianTokenizer


# Define a function to modify hidden states
def your_input_modification(hidden_states):
    # Example modification: no-op
    mod = hidden_states
    return mod


# Custom Layer Wrapper for MarianMT
class CustomLayerWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # Modify hidden_states before passing them to the layer
        modified_hidden_states = your_input_modification(hidden_states)

        # Ensure that attention_mask and any other necessary arguments are forwarded
        return self.layer(modified_hidden_states, attention_mask, **kwargs)



# Load the model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-he"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Example: Modify the first encoder layer
nlayer = 0  # Specify the layer index you want to modify
original_layer = model.model.encoder.layers[nlayer]
wrapped_layer = CustomLayerWrapper(original_layer)
model.model.encoder.layers[nlayer] = wrapped_layer

# Example usage
inputs = tokenizer("Hello", return_tensors="pt")

decoder_start_token_id = tokenizer.pad_token_id  # or another token like bos_token_id
decoder_input_ids = torch.full((inputs.input_ids.size(0), 1), decoder_start_token_id, dtype=torch.long).to(inputs.input_ids.device)

# Custom model call (example)
outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_input_ids, output_hidden_states=True)

generated_ids = model.generate(inputs.input_ids)

generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("Generated Text: ", generated_text)

