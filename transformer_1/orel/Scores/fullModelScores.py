from Scores.modelTester import test
import joblib


dataset_path = 'wikipedia_test_data.csv'
model_path = "transformer_1/orel/pretrainedModels/models/15Tokens/model_wiki_30211_30210_new_none_2words_learning.pkl"

model = joblib.load(model_path)

test(dataset_path, "full",model)