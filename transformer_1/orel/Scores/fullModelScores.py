# from modelTester import test
# import joblib

# import sys
# import os.path

# sys.path.append(os.path.dirname(os.path.realpath("orel")))

# print(sys.path)

# dataset_path = 'wikipedia_test_data.csv'


# model = joblib.load(f'/home/ddn1/Documents/GitHub/HebrewLLM/transformer_1/orel/pretrainedModels/models/15Tokens/model_wiki_30211_30210_new_none_2words_learning.pkl')


# test(dataset_path, "full",model)


import joblib
import sys
import os

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
    print("CombinedModel imported successfully.")
except ImportError as e:
    print(f"Error importing CombinedModel: {e}")
    sys.exit(1)  # Exit if the import fails

dataset_path = os.path.join(project_dir, 'wikipedia_test_data.csv')
model_dir = os.path.join(model_dir, 'orel/pretrainedModels/models/15Tokens')
model_path = os.path.join(model_dir, 'model_wiki_30211_30210_new_none_2words_learning.pkl')
# model_path = os.path.join(model_dir, 'model_wiki_30211_30210_new_none_5words_learning.pkl')
print(f"model_path: {model_path}")
# print(f"model_path: {model_path}")

# Verify if the model file exists
if not os.path.exists(model_path):
    print(f"Model file does not exist: {model_path}")
    sys.exit(1)

# Print contents of the directory for verification
print(f"Contents of model_dir: {os.listdir(model_dir)}")

model = joblib.load(model_path)
print(f"Model loaded successfully. {type(model)}")


from modelTester import test

test(
    hebrew_dataset_path=dataset_path, 
    model_type="full", 
    model=model,
    input_size=2
    )
