import os
from torch import nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

os.environ['CUDA_VISIBLE_DEVICES'] = ''

model_name = "Helsinki-NLP/opus-mt-en-he"
device = "cpu"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def your_input_modification(encoder_hidden_states):
    # Implement your modification logic here if needed
    modified_states = encoder_hidden_states + 0.9

    return modified_states

class CustomLayerWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, **kwargs):
        # Apply modifications to hidden_states and attention_mask here
        modified_hidden_states = your_input_modification(encoder_hidden_states)

        # Pass modified_hidden_states and modified_attention_mask to the original layer
        return self.layer(hidden_states, attention_mask=attention_mask,
                          encoder_hidden_states=modified_hidden_states,
                          encoder_attention_mask=encoder_attention_mask,
                          **kwargs)

def translator_activation_different_layer(encoder_hidden_states, nlayer):
    # Create an instance of the layer you want to modify
    custom_layer = model.model.decoder.layers[nlayer]
    # Wrap the layer inside the custom wrapper
    wrapped_layer = CustomLayerWrapper(custom_layer)
    # Replace the layer with the wrapped layer
    model.model.decoder.layers[nlayer] = wrapped_layer

    # Create a dummy input
    inputs = tokenizer.encode("Hello", return_tensors="pt").to(device)

    # Generate output
    outputs = model.generate(inputs, output_hidden_states=True,
                             return_dict_in_generate=True, max_length=512)

    # Access the generated token IDs
    token_ids = outputs.sequences
    # Decode the token IDs using the tokenizer
    generated_text = tokenizer.decode(token_ids[0], skip_special_tokens=True)

    return outputs, generated_text



if __name__ == '__main__':
    tokenized_sentence = tokenizer.encode("Hello", return_tensors='pt').to(device)
    translated_tokens = model.generate(tokenized_sentence, output_hidden_states=True,
                                       return_dict_in_generate=True, max_length=512)
    encoder_hidden_states = translated_tokens.encoder_hidden_states[0]

    layer_to_modify = 1
    outputs, generated_text = translator_activation_different_layer(encoder_hidden_states=encoder_hidden_states, nlayer=layer_to_modify)

    # Print the generated text
    print("Generated Text: ", generated_text)
