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
from transformers import MarianMTModel, MarianTokenizer, MarianConfig
import torch
from torch import nn

def your_input_modification(hidden_states):
    modified_states = hidden_states
    return modified_states

class CustomMarianMTModel(MarianMTModel):
    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, encoder_outputs=None,
                past_key_values=None, inputs_embeds=None, decoder_inputs_embeds=None, **kwargs):

        modified_hidden_states = your_input_modification(inputs_embeds)

        return super().forward(input_ids=input_ids, attention_mask=attention_mask,
                               decoder_input_ids=decoder_input_ids, encoder_outputs=encoder_outputs,
                               past_key_values=past_key_values, inputs_embeds=modified_hidden_states,
                               decoder_inputs_embeds=decoder_inputs_embeds, **kwargs)

# Load the tokenizer and model
model_name = "Helsinki-NLP/opus-mt-en-he"
tokenizer = MarianTokenizer.from_pretrained(model_name)
config = MarianConfig.from_pretrained(model_name)
custom_model = CustomMarianMTModel.from_pretrained(model_name, config=config)
model = MarianMTModel.from_pretrained(model_name)

# Encode the input text
input_text = "Talia"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate initial outputs
outputs = model.generate(input_ids=input_ids, output_hidden_states=True, max_length=512, return_dict_in_generate=True)

generated_text = tokenizer.decode(outputs[0][0], skip_special_tokens=True)
print("Generated:", generated_text)

# Access the hidden states from the encoder
layer = 0
hidden_states = outputs.encoder_hidden_states[layer]

# Use the hidden states as input for another forward pass
new_outputs = custom_model.generate(inputs_embeds=hidden_states, output_hidden_states=True, max_length=512, return_dict_in_generate=True)

# Decode and print the generated text
generated_text = tokenizer.decode(new_outputs[0][0], skip_special_tokens=True)
print("Custom Generated:", generated_text)
#-----------------------------------------------------------------------------------------------------------
# from transformers import MarianMTModel, MarianTokenizer, MarianConfig
# import torch
#
#
# class CustomMarianModel(MarianMTModel):
#     def __init__(self, config):
#         super().__init__(config)
#
#     def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
#                 encoder_outputs=None, past_key_values=None, inputs_embeds=None,
#                 decoder_inputs_embeds=None, use_cache=None, output_attentions=None,
#                 output_hidden_states=None, return_dict=None, **kwargs):
#
#         # Run through the encoder if encoder_outputs are not provided
#         if encoder_outputs is None:
#             encoder_outputs = self.model.encoder(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#             )
#
#         # Ensure decoder_input_ids are provided for the decoder
#         if decoder_input_ids is None:
#             # Assuming using the start token ID as the first decoder input ID for generation
#             # This is a simplification; in practice, you might need to dynamically determine this
#             start_token_id = self.config.decoder_start_token_id
#             if start_token_id is None:
#                 raise ValueError("No decoder start token id found in the model config")
#             # Create a tensor of shape (batch_size, 1) filled with the start token ID
#             decoder_input_ids = torch.full((input_ids.shape[0], 1), start_token_id, dtype=torch.long).to(
#                 input_ids.device)
#
#         # Decoder forward pass
#         decoder_outputs = self.model.decoder(
#             input_ids=decoder_input_ids,
#             attention_mask=attention_mask,
#             encoder_hidden_states=encoder_outputs[0],
#             encoder_attention_mask=attention_mask,
#             past_key_values=past_key_values,
#             inputs_embeds=decoder_inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             **kwargs
#         )
#
#         return decoder_outputs
#
#
# # Load the model and tokenizer
# model_name = "Helsinki-NLP/opus-mt-en-he"
# tokenizer = MarianTokenizer.from_pretrained(model_name)
# model_config = MarianConfig.from_pretrained(model_name)
# custom_model = CustomMarianModel.from_pretrained(model_name, config=model_config)
#
# # Encode the input text and generate output
# input_text = "Hello, world!"
# input_ids = tokenizer.encode(input_text, return_tensors="pt")
# decoder_start_token_id = model_config.decoder_start_token_id
# decoder_input_ids = torch.tensor([[decoder_start_token_id]])  # Assuming you're generating from start token
#
# outputs = custom_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
# generated_ids = custom_model.generate(input_ids)
#
# print("Generated:", tokenizer.decode(generated_ids[0], skip_special_tokens=True))

