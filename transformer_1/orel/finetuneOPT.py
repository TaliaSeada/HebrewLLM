import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer, OPTForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Model and tokenizer initialization
llm_model_name = "facebook/opt-350m"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm = OPTForCausalLM.from_pretrained(llm_model_name)

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('wikipedia_data_15.csv')

# Encode labels
label_encoder = LabelEncoder()
df['labels'] = label_encoder.fit_transform(df['label'])
df = df.drop(columns=['label'])

# Convert the DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Tokenize the dataset with padding and truncation
def tokenize_function(examples):
    return llm_tokenizer(examples['Hebrew sentence'], padding='max_length', truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["Hebrew sentence"])

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=llm_tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Trainer setup
trainer = Trainer(
    model=llm,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets,
)

# Train the model
trainer.train()

# Save the finetuned model and tokenizer with a new name
finetuned_model_dir = "./finetuned_opt_350m_custom"
trainer.save_model(finetuned_model_dir)
llm_tokenizer.save_pretrained(finetuned_model_dir)
