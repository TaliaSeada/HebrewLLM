import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, MarianTokenizer, MarianMTModel, AutoTokenizer, OPTForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import pandas as pd
from torch.utils.data import DataLoader
from data.dataManipulation import pad, pad_and_mask
from model.HiddenStateTransformer import HiddenStateTransformer, HiddenStateTransformer2, train_model, test_model
from generalTransformer import CustomLayerWrapper, CustomLayerWrapper2
import torch.nn.functional as F
import math

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class CombinedModel(nn.Module):
    def __init__(self, tokenizer1, translator1, transformer1, llm_tokenizer, llm, transformer2, tokenizer2, translator2):
        super(CombinedModel, self).__init__()
        self.tokenizer1 = tokenizer1
        self.translator1 = translator1
        
        self.transformer1 = transformer1
        
        self.llm_tokenizer = llm_tokenizer
        self.llm: OPTForCausalLM = llm
        
        self.transformer2 = transformer2
        
        self.tokenizer2 = tokenizer2
        self.translator2: MarianMTModel = translator2

        # Freeze parameters as before
        for param in self.translator1.parameters():
            param.requires_grad = False
            
        for param in self.llm.parameters():
            param.requires_grad = False
    
        for param in self.translator2.parameters():
            param.requires_grad = False
            
        original_layer = self.llm.base_model.decoder.layers[1]
        wrapped_layer = CustomLayerWrapper(original_layer, None)
        self.llm.base_model.decoder.layers[1] = wrapped_layer
        
        original_layer2 = self.translator2.model.encoder.layers[1]
        wrapped_layer2 = CustomLayerWrapper2(original_layer2, None)
        self.translator2.model.encoder.layers[1] = wrapped_layer2

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Ensure input_ids and attention_mask are on the correct device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Process input_ids through the model
        text = self.tokenizer1.decode(input_ids[0].cpu().numpy())  # Decode the input ids to text
        
        x, _ = self.hebrew_to_input(text)
        if x.shape[1] > 15:
            return None

        x = self.transformer1(x.to(device))
        inputs = self.llm_tokenizer(" " * 14, return_tensors="pt").to(device)
        self.inject_layer(layer_hs=x, layer_num=1, name="decoder")
        outputs = self.llm(**inputs, output_hidden_states=True)
        llm_last_hidden_state = outputs.hidden_states[-1]
        x = self.transformer2(llm_last_hidden_state.to(device))
        self.inject_layer(layer_hs=x, layer_num=1, name="encoder")
        logits = self.generate_predicted_distribution(text)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1).to(device))
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


    # The rest of the methods remain unchanged

    def inject_layer(self, layer_hs, layer_num, name="decoder"):

        # Replace the layer with the wrapped layer
        if name == "decoder":
            self.llm.base_model.decoder.layers[layer_num].hs = layer_hs
        else:
            self.translator2.model.encoder.layers[layer_num].hs = layer_hs


    def hebrew_to_input(self,h_text):
        # Translator
        inputs = self.tokenizer1(h_text, return_tensors="pt")

        # Encode the source text
        # generated_ids = self.translator1.generate(inputs.input_ids)
        generated_ids = self.translator1(**inputs)

        # print(f"Hebrew input ids = {len(generated_ids[0])}")

        clean_token_num = len(generated_ids[0]) - 2

        # for t in generated_ids:
        #     print(f"translated token1 {t} = {hebrew_translator_tokenizer.convert_ids_to_tokens(t)}")

        # Append hidden states
        translator_outputs = self.translator1(input_ids=inputs.input_ids, decoder_input_ids=generated_ids,
                                                    output_hidden_states=True)

        # Extract the last hidden state from translator
        translator_last_hidden_state = translator_outputs.decoder_hidden_states[-1]

        data = [(pad(translator_last_hidden_state),)]
        data_padded, labels_padded, data_masks, labels_masks = pad_and_mask(data, False)

        return data_padded, clean_token_num


    def generate_predicted_distribution(self, text):
        
        # Get the tokens for the target sentence
        known_target_ids = self.tokenizer2(text_target=text, return_tensors="pt").input_ids

        # Create a tensor filled with pad token IDs, with the desired length
        max_length = 15

            
        # # Trick Translator by giving it a dummy that contain the desired number of tokens (In our case 15)
        # # and replace the first layer as it got other word embedding.
        inputs = self.tokenizer2("a " * 14, return_tensors="pt")
        
        
        # Ensure the attention mask is correctly shaped
        attention_mask = torch.ones((1, 15))
        inputs['attention_mask'] = attention_mask
        
        decoder_len = max((max_length - known_target_ids.shape[1]), 0)
        
        # Prepare decoder_input_ids, starting with the <pad> token
        decoder_input_ids = torch.full(
            (inputs.input_ids.size(0),  decoder_len), self.tokenizer2.pad_token_id, dtype=torch.long
        )

        # Concatenate with input_ids shifted right
        decoder_input_ids = torch.cat([known_target_ids, decoder_input_ids], dim=1)

        # print(f"decoder_input_ids = {decoder_input_ids},\nShape = {decoder_input_ids.shape}")

        # Forward pass to get the logits
        outputs = self.translator2(
            input_ids=inputs.input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True
        )
                
        # Getting the logits (usually the last hidden state contains the logits for token predictions)
        logits = outputs.logits
        
        q = logits

        return q






lr=0.006334926670051613
train_size = 30210


# Hebrew to english translator
He_En_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
He_En_tokenizer = MarianTokenizer.from_pretrained(He_En_model_name)
He_En_translator_model = MarianMTModel.from_pretrained(He_En_model_name)

# Transformer 1
t1 = joblib.load('/home/ddn1/Documents/GitHub/HebrewLLM/transformer_1/orel/pretrainedModels/models/15Tokens/model_wiki_10414_36000.pkl')

# LLM model
llm_model_name = "facebook/opt-350m"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm = OPTForCausalLM.from_pretrained(llm_model_name)

# Transformer 2
t2 = HiddenStateTransformer2(input_size=512,output_size=512, num_layers=1, num_heads=2, dim_feedforward=256, dropout=0.15)

# English to Hebrew translator
En_He_model_name = "Helsinki-NLP/opus-mt-en-he"
En_He_tokenizer = MarianTokenizer.from_pretrained(En_He_model_name)
En_He_translator_model = MarianMTModel.from_pretrained(En_He_model_name)


# Model initialization
model = CombinedModel(He_En_tokenizer, He_En_translator_model, t1, llm_tokenizer, llm, t2, En_He_tokenizer, En_He_translator_model)

# Tokenize dataset (assuming you have a DataFrame `df` as in your example)
def tokenize_function(examples):
    return He_En_tokenizer(examples['Hebrew sentence'], padding='max_length', truncation=True, max_length=511)
    # return He_En_tokenizer(examples['Hebrew sentence'], padding='max_length', truncation=True)


# Load the CSV file into a pandas DataFrame
df = pd.read_csv('wikipedia_data_15.csv')

# Encode labels
label_encoder = LabelEncoder()
df['labels'] = label_encoder.fit_transform(df['label'])
df = df.drop(columns=['label'])

# Convert the DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(df)


tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["Hebrew sentence"])

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=He_En_tokenizer, mlm=False)


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=500,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,  # Assuming you use a similar data collator as before
)

# Train the model
trainer.train()

# Save the finetuned model
finetuned_model_dir = "./improved_combined_model"
trainer.save_model(finetuned_model_dir)
