import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModel, AutoTokenizer, OPTForCausalLM
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from datasets import load_dataset

# build models
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "facebook/opt-350m"
model = OPTForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load your dataset
dataset = load_dataset('C:\\Users\\relwe\\OneDrive\\Documents\\GitHub\\HebrewLLM\\wikipedia_data.csv')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=15)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

a=0