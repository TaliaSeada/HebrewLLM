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
import numpy as np

EPOCHS = 2

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


def find_best_hypers(trial, dataset_path: str, stop_index, epochs, test_size):

    # Define the hyperparameters to tune
    # lrRange = np.arange(0.0001,0.01,0.0005)
    lr = trial.suggest_float('lr', 1e-6, 1e-1, log=True)

    
    # for lr in lrRange:
    
    t1 = joblib.load('C:\\Users\\relwe\\Documents\\HebrewLLM\\transformer_1\\orel\\pretrainedModels\\models\\10Tokens\\general_model.pkl')

    t2 = joblib.load('C:\\Users\\relwe\\Documents\\HebrewLLM\\transformer_2\\pretranedModels\\models\\15Tokens\\model_15_tokens_talia.pkl')
    
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
    
    combined_model.train()

    optimizer = optim.Adam(combined_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    
    for epoch in range(epochs):
        # Train the model
        train_loss = train_combined_model(dataset_path, stop_index, combined_model, criterion, optimizer)
        
        prms = [param for param in combined_model.transformer1.parameters()]
        
        validation_loss = test_combined_model(dataset_path, stop_index, test_size, combined_model, criterion, optimizer)
        
        print(f"Epoch {epoch+1}, Training Loss: {train_loss}, Validation Loss: {validation_loss:.4f}")
        # print(f"Epoch {epoch+1}, Training Loss: {train_loss}")

        # Report intermediate objective value.
        trial.report(validation_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            print("/n/n==========its because of me!!==========/n/n", flush=True)
            raise optuna.exceptions.TrialPruned()\
        # Validation phase
        # combined_model.eval()
        
    print(f"Validation loss: {validation_loss}")

    return validation_loss


def findBest(dataset_path: str, stop_index, epochs, test_size):
     # Define the objective function
    def objective(trial):
        return find_best_hypers(trial, dataset_path, stop_index, epochs, test_size)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)


    # Print best hyperparameters
    print(study.best_params)


def train_combined_model(dataset_path, stop_index, model: CombinedModel, criterion, optimizer,):
    df = pd.read_csv(dataset_path)
    
    # Train the model
    # model.train()  # Set the model to training mode
    train_loss = 0
    counter = 0
    
    for index, row in df.iterrows():
        if index > stop_index:
            break
        hebrew_sentence = row['Hebrew sentence']
        target_hebrew_sentence = row['Hebrew sentence'] + " " + row['label']
        
        # with torch.enable_grad():
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
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"Parameter: {name}, Grad: {param.grad}")
        
        
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"Parameter: {name}, Grad: {param.grad}")
        optimizer.step()
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"Parameter: {name}, Grad: {param.grad}")

        train_loss += loss.item()

    train_loss /= counter
    return train_loss
    
    
def test_combined_model(dataset_path, start_index, stop_index, model: CombinedModel, criterion, optimizer):
    df = pd.read_csv(dataset_path)
    
    # model.eval()
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
            
            # actual.requires_grad_()
            
            loss = criterion(actual, expected)

            # # Back Propagation
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            test_loss += loss.item()

    test_loss /= counter
    return test_loss
    

dataset_path = 'C:\\Users\\relwe\\Documents\\HebrewLLM\\wikipedia_data.csv'

findBest(dataset_path, 2, EPOCHS, 1)