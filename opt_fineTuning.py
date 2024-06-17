import torch
from transformers import OPTForCausalLM, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

# build models
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load your dataset
dataset = load_dataset('csv', data_files='C:\\Users\\relwe\\OneDrive\\Documents\\GitHub\\HebrewLLM\\wikipedia_data.csv')
# Function to merge 'sentence' and 'label' into one column
def merge_columns(examples):
    examples['sentence'] = examples['sentence'] + " " + examples['label']
    return examples

# Apply the function to the dataset
dataset = dataset.map(merge_columns)

# Remove the 'label' column as it's now included in 'sentence'
dataset = dataset.remove_columns("label")

# Tokenize the dataset
def tokenize_function(examples):
    # Tokenize the sentence
    inputs = tokenizer(examples['sentence'], padding="max_length", truncation=True, max_length=100)
    # # Tokenize the expected next word
    # with tokenizer.as_target_tokenizer():
    #     labels = tokenizer(examples['label'], padding="max_length", truncation=True, max_length=22)

    # # Ensure labels are properly formatted for causal language modeling
    # inputs["labels"] = labels["input_ids"]
    return inputs

# Map the tokenize function to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# # Split dataset into train and validation
# tokenized_datasets = tokenized_datasets["train"].train_test_split(test_size=0.1)
train_dataset = tokenized_datasets["train"]
# eval_dataset = tokenized_datasets["test"]
# Prepare data loaders
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Hyper parameters
batch_size = 8
lr = 2e-5
num_epochs = 3
weight_decay = 0.01

train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    # evaluation_strategy="epoch",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    # per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=weight_decay,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
path_to_save_finetuned_model = "C:\\Users\\relwe\\OneDrive\\Documents\\GitHub\\HebrewLLM\\finetuned_model.pth"
model.save_pretrained(path_to_save_finetuned_model)
tokenizer.save_pretrained(path_to_save_finetuned_model)