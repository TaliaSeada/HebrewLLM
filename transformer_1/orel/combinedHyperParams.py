import optuna
import torch
from combinedTransformersModel import CombinedModel
import torch.optim as optim
import torch.nn as nn
import joblib
from transformers import AutoTokenizer, AutoModel, MarianTokenizer, MarianMTModel, AutoTokenizer, OPTForCausalLM
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from data.dataManipulation import pad, pad_and_mask
from model.HiddenStateTransformer import HiddenStateTransformer, HiddenStateTransformer2, train_model, test_model
from generalTransformer import CustomLayerWrapper, CustomLayerWrapper2
import torch.nn.functional as F
import math

EPOCHS = 5

# Hebrew to english translator
He_En_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
He_En_tokenizer = MarianTokenizer.from_pretrained(He_En_model_name)
He_En_translator_model = MarianMTModel.from_pretrained(He_En_model_name)


# LLM model
llm_model_name = "facebook/opt-350m"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm = OPTForCausalLM.from_pretrained(llm_model_name)

# English to Hebrew translator
En_He_model_name = "Helsinki-NLP/opus-mt-en-he"
En_He_tokenizer = MarianTokenizer.from_pretrained(En_He_model_name)
En_He_translator_model = MarianMTModel.from_pretrained(En_He_model_name)


def find_best_hypers(criterion, trial, dataset_path: str, stop_index, epochs, test_size):

    # Define the hyperparameters to tune
    # lr = trial.suggest_float('lr', 5e-4, 5e-3, log=True)
    lr = trial.suggest_categorical('lr', [5e-4, 3e-4, 1e-4])


    t1 = joblib.load('transformer_1/orel/pretrainedModels/models/10Tokens/general_model.pkl')

    t2 = joblib.load('C:\\Users\\orelz\\OneDrive\\שולחן העבודה\\work\\Ariel\\HebrewLLM\\transformer_2\\pretranedModels\\models\\15Tokens\\model_15_tokens_talia.pkl')
    
    combined_model = None
    
    # Create the model, criterion, and optimizer
    combined_model = CombinedModel(tokenizer1=He_En_tokenizer,
                                translator1=He_En_translator_model,
                                transformer1=t1,
                                llm_tokenizer=llm_tokenizer,
                                llm=llm,
                                transformer2=t2,
                                tokenizer2=En_He_tokenizer,
                                translator2=En_He_translator_model
                                )

    optimizer = optim.Adam(combined_model.parameters(), lr=lr)
    # optimizer = optim.AdamW(combined_model.parameters(), lr=lr)
    # optimizer = optim.SGD(combined_model.parameters(), lr=lr)
    # optimizer = optim.RAdam(combined_model.parameters(), lr=lr)

    
    for epoch in range(epochs):
        # Train the model
        train_loss = train_combined_model(dataset_path, stop_index, combined_model, criterion, optimizer)

        # Validation phase
        validation_loss = test_combined_model(dataset_path, stop_index, test_size, combined_model, criterion, optimizer)
        
        print(f"Epoch {epoch+1}, Training Loss: {train_loss}, Validation Loss: {validation_loss:.4f}")

        # Report intermediate objective value.
        trial.report(validation_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            print("/n/n==========its because of me!!==========/n/n", flush=True)
            raise optuna.exceptions.TrialPruned()

    return validation_loss


def findBest(criterion, dataset_path: str, stop_index, epochs, test_size):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: find_best_hypers(criterion,trial,dataset_path,stop_index,epochs,test_size), n_trials=5)

    # Print best hyperparameters
    print(study.best_params)


def train_combined_model(dataset_path, stop_index, model: CombinedModel, criterion, optimizer,):
    df = pd.read_csv(dataset_path)
    
    # Train the model
    model.train()  # Set the model to training mode
    train_loss = 0
    counter = 0
    
    for index, row in df.iterrows():
        if index > stop_index:
            break
        hebrew_sentence = row['Hebrew sentence']
        target_hebrew_sentence = row['Hebrew sentence'] + " " + row['label']
        
        # Outputs predicted distribution for each token
        q = model(hebrew_sentence)
        
        if q is None:
            continue
        
        counter += 1

        # Get the tokens for the target sentence
        target_ids = En_He_tokenizer(text_target=target_hebrew_sentence, return_tensors="pt")
        
        max_left = q[0, 1:, :].shape[0]
        max_right = target_ids.input_ids.squeeze(0).shape[0]

        desired_len = min(max_left, max_right, 14)
        
        actual = q[0,1:desired_len + 1, :]
        expected = target_ids.input_ids.squeeze(0)[:desired_len]
        
        actual.requires_grad_()
        
        loss = criterion(actual, expected)

        # Back Propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    if counter > 0:
        train_loss /= counter
    return train_loss
    
    
def test_combined_model(dataset_path, start_index, stop_index, model: CombinedModel, criterion, optimizer):
    df = pd.read_csv(dataset_path)
    
    # Train the model
    model.eval()  # Set the model to training mode
    test_loss = 0
    counter = 0
    
    for index, row in df.iterrows():
        if index < start_index:
            continue
        if index > start_index + stop_index:
            break
        hebrew_sentence = row['Hebrew sentence']
        target_hebrew_sentence = row['Hebrew sentence'] + " " + row['label']
        
        with torch.no_grad():  # No gradient calculation
            # Outputs predicted distribution for each token
            q = model(hebrew_sentence)
            
            if q is None:
                continue
            
            counter += 1

            # Get the tokens for the target sentence
            target_ids = En_He_tokenizer(text_target=target_hebrew_sentence, return_tensors="pt")
            
            max_left = q[0, 1:, :].shape[0]
            max_right = target_ids.input_ids.squeeze(0).shape[0]

            desired_len = min(max_left, max_right, 14)
            
            actual = q[0,1:desired_len + 1, :]
            expected = target_ids.input_ids.squeeze(0)[:desired_len]
            
            actual.requires_grad_()
            
            loss = criterion(actual, expected)

            test_loss += loss.item()
    if counter > 0:
        test_loss /= counter
    return test_loss
    

criterion = nn.CrossEntropyLoss()
dataset_path = 'transformer_1/orel/sampled_data.csv'

findBest(criterion, dataset_path, 100, 3, 100)