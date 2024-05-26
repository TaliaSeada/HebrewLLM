import TransGeneralEmbeddingToOPT as ge
import torch
import torch.nn as nn


def findMean(datasetPath: str, criterion):
    # Load dataset
    loaded_data = torch.load(datasetPath)
    
    # Pad data to a specific length
    data = [ge.pad(details[0]) for _, details in loaded_data.items()]
    labels = [ge.pad(details[1]) for _, details in loaded_data.items()]

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
findMean("resources/datasets/up_to_ten_tokens_dataset_wiki_5.pt", criterion)