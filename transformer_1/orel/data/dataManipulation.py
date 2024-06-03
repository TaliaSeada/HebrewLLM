import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from model.config import MAX_TANSOR_LENGTH,BATCH_SIZE,TEST_SIZE,VALIDATION_SIZE


def pad_and_mask(batch, pad_labels = True, max_tansor_length: int = MAX_TANSOR_LENGTH):

    data = [item[0].squeeze(0) for item in batch]
    
    # Pad data to the maximum length in the batch same for labels below
    data_padded = pad_sequence(data, batch_first=True, padding_value=0)
    
    #  Calculate how many zero vectors are needed to padd to MAX_TANSOR_LENGTH
    data_padding_num = max_tansor_length - data_padded.shape[1]
    
    # Check if we need to add any vectors
    if data_padding_num > 0:
        # Create a tensor of zero vectors with the same batch size and vector size, on the same device
        zero_vector_data = torch.zeros(data_padded.shape[0], data_padding_num, data_padded.shape[2], device=data_padded.device)
        
        # Concatenate the original tensor with the zero vectors along dimension 1
        data_padded = torch.cat([data_padded, zero_vector_data], dim=1)
        
    # Create masks for data same for labels
    data_masks = (data_padded != 0).any(dim=-1).float()
    
    labels_padded = None
    labels_masks = None
    
    if pad_labels:
        labels = [item[1].squeeze(0) for item in batch]

        labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
        
        lables_padding_num = max_tansor_length - labels_padded.shape[1]


        if lables_padding_num > 0:
            # Create a tensor of zero vectors with the same batch size and vector size, on the same device
            zero_vector_labels = torch.zeros(labels_padded.shape[0], lables_padding_num, labels_padded.shape[2], device=labels_padded.device)
            
            # Concatenate the original tensor with the zero vectors along dimension 1
            labels_padded = torch.cat([labels_padded, zero_vector_labels], dim=1)

        labels_masks = (labels_padded != 0).any(dim=-1).float()
    
    return data_padded, labels_padded, data_masks, labels_masks


def pad(data, max_tansor_length: int = MAX_TANSOR_LENGTH):
    data_padding_num = max_tansor_length - data.shape[1]

    # Check if we need to add any vectors
    if data_padding_num > 0:
        # Create a tensor of zero vectors with the same batch size and vector size, on the same device
        zero_vector_data = torch.zeros(data.shape[0], data_padding_num, data.shape[2], device=data.device)
        
        # Concatenate the original tensor with the zero vectors along dimension 1
        data = torch.cat([data, zero_vector_data], dim=1)
    return data


def create_data_loaders(dataset_path: str, batch_size=BATCH_SIZE) -> tuple:
    
    print(f"Dataset = {dataset_path}")
    
    # Load dataset
    loaded_data = torch.load(dataset_path)
    
    data = []
    labels = []
    
    i = 0
    for _, details in loaded_data.items():
        # Decides How much data to train on
        # if i > 100:
        #     break
        data.append(pad(details[0]))
        labels.append(pad(details[1]))
        i += 1

    # Convert lists of tensors to single tensors by stacking
    data = torch.stack(data)  # Stacks tensors along a new dimension
    labels = torch.stack(labels)
    
    # data = data.unsqueeze(-1)
    
    # First, split into training and temp
    data_train, data_temp, labels_train, labels_temp = train_test_split(data, labels,
                                                                        test_size=TEST_SIZE + VALIDATION_SIZE,
                                                                        random_state=42)

    # Now split temp into validation and test
    data_val, data_test, labels_val, labels_test = train_test_split(data_temp, labels_temp, 
                                                                    test_size=(VALIDATION_SIZE/(TEST_SIZE + VALIDATION_SIZE)),
                                                                    random_state=42)
    
    # Convert data to PyTorch tensors and wrap them in a dataset
    train_dataset = TensorDataset(data_train, labels_train)    
    val_dataset = TensorDataset(data_val, labels_val)
    test_dataset = TensorDataset(data_test, labels_test)


    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_and_mask)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_and_mask)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_and_mask)
    
    return train_loader, val_loader, test_loader