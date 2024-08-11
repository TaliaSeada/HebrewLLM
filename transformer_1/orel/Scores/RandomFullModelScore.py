import joblib
import sys
import os
from modelTester import test
from transformers import AutoTokenizer, AutoModel, MarianTokenizer, MarianMTModel, AutoTokenizer, OPTForCausalLM

# Add the directory containing combinedTransformersModel.py to the Python path
project_dir = '/home/ddn1/Documents/GitHub/HebrewLLM/'
model_dir = os.path.join(project_dir, 'transformer_1')
sys.path.append(model_dir)
orel_folder_path = os.path.join(model_dir, 'orel')
sys.path.append(orel_folder_path)

print(f"sys.path: {sys.path}")

# Import CombinedModel from combinedTransformersModel.py
try:
    from combinedTransformersModel import CombinedModel
    from model.HiddenStateTransformer import HiddenStateTransformer, HiddenStateTransformer2, train_model, test_model

    print("CombinedModel imported successfully.")
except ImportError as e:
    print(f"Error importing CombinedModel: {e}")
    sys.exit(1)  # Exit if the import fails


# Hebrew to english translator
He_En_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
He_En_tokenizer = MarianTokenizer.from_pretrained(He_En_model_name)
He_En_translator_model = MarianMTModel.from_pretrained(He_En_model_name)

# Transformer 1
# t1 = HiddenStateTransformer(input_size=1024,output_size=1024, num_layers=1, num_heads=4, dim_feedforward=256, dropout=0.25)
t1 = joblib.load('/home/ddn1/Documents/GitHub/HebrewLLM/transformer_1/orel/pretrainedModels/models/15Tokens/model_wiki_10414_36000.pkl')


# LLM model
llm_model_name = "facebook/opt-350m"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm = OPTForCausalLM.from_pretrained(llm_model_name)

# Transformer 2
t2 = HiddenStateTransformer2(input_size=512,output_size=512, num_layers=1, num_heads=2, dim_feedforward=256, dropout=0.15)
# t2 = joblib.load('/home/ddn1/Documents/GitHub/HebrewLLM/transformer_2/pretrainedModels/models/15Tokens/model_15_tokens_talia.pkl')


# English to Hebrew translator
En_He_model_name = "Helsinki-NLP/opus-mt-en-he"
En_He_tokenizer = MarianTokenizer.from_pretrained(En_He_model_name)
En_He_translator_model = MarianMTModel.from_pretrained(En_He_model_name)


model = CombinedModel(tokenizer1=He_En_tokenizer,
                               translator1=He_En_translator_model,
                               transformer1=t1,
                               llm_tokenizer=llm_tokenizer,
                               llm=llm,
                               transformer2=t2,
                               tokenizer2=En_He_tokenizer,
                               translator2=En_He_translator_model
                               )

dataset_path = os.path.join(project_dir, 'wikipedia_test_data.csv')

test(
    hebrew_dataset_path=dataset_path,
    model_type="full",
    model=model,
    input_size=2
)
