from transformers import AutoTokenizer, OPTForCausalLM

# Load the fine-tuned model and tokenizer
finetuned_model_dir = "./finetuned_opt_350m_custom"
tokenizer = AutoTokenizer.from_pretrained(finetuned_model_dir)
model = OPTForCausalLM.from_pretrained(finetuned_model_dir)


from modelTester import test


path = 'wikipedia_test_data.csv'

test(
    hebrew_dataset_path=path, 
    model_type="finetune", 
    model=model, 
    tokenizer=tokenizer
    )
