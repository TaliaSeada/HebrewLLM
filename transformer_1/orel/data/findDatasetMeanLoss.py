
# import os
# import sys
from data.dataManipulation import pad
# from dataManipulation import pad

# # Add the parent directory to sys.path
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if parent_dir not in sys.path:
#     sys.path.insert(0, parent_dir)


import torch
import torch.nn as nn


def findMean(datasetPath: str, criterion):
    # Load dataset
    loaded_data = torch.load(datasetPath)
    
    # Pad data to a specific length
    data = [pad(details[0]) for _, details in loaded_data.items()]
    labels = [pad(details[1]) for _, details in loaded_data.items()]

    # Convert lists of tensors to single tensors by stacking
    data = torch.stack(data)  # Stacks tensors along a new dimension
    labels = torch.stack(labels)

    # Calc mean
    mean_values = torch.mean(data, dim=0)

    # Expand the mean values back to the original shape
    avg_hs = mean_values.unsqueeze(0).expand_as(data)
    loss = criterion(avg_hs, data)
    print(f"Mean loss = {loss.item()}")
    print(f"Dataset shape = {data.shape}")



criterion = nn.MSELoss()
# findMean("resources/datasets/up_to_ten_tokens_dataset.pt", criterion)
# findMean("resources/datasets/dataset_wiki_up_to_15_tokens.pt", criterion)
findMean('resources/datasets/dataset_wiki_up_to_15_tokens_36000.pt', criterion)