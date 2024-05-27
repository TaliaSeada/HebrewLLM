import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import torch.optim as optim
import optuna
from torch.nn.utils.rnn import pad_sequence


MAX_TANSOR_LENGTH = 15
BATCH_SIZE = 64
TEST_SIZE = 0.10
VALIDATION_SIZE = 0.15
EPOCHS = 5
DROPOUT = 0.1


# Transformer
class HiddenStateTransformer(nn.Module):
    def __init__(self, input_size=1024, output_size=MAX_TANSOR_LENGTH, num_layers=2, num_heads=1, dim_feedforward=128, dropout=DROPOUT, activation=F.relu):
        super(HiddenStateTransformer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        # self.input_transform = nn.Linear(1024, output_size)

        # Transformer Encoder Layer
        encoder_layers = TransformerEncoderLayer(d_model=input_size, nhead=num_heads,
                                                 dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
        encoder_layers.self_attn.batch_first = True
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # # Linear layer to map to the target hidden size
        # self.fc = nn.Linear(input_size, output_size)

    def forward(self, src, src_key_padding_mask=None):
        # src = self.input_transform(src)
        # src shape: (batch_size, seq_length, input_size)
        encoded = self.transformer_encoder(src)
        # encoded shape: (batch_size, seq_length, input_size)
        # output = self.fc(encoded)
        output = encoded
        # output shape: (batch_size, seq_length, output_size)
        return output


def pad_and_mask(batch, max_tansor_length: int = MAX_TANSOR_LENGTH):
    data = [item[0].squeeze(0) for item in batch]
    labels = [item[1].squeeze(0) for item in batch]

    # Calculate max length for padding
    max_length = max(max(seq.size(0) for seq in data), max(seq.size(0) for seq in labels))
    # print(max_length)


    # Pad data and labels to the maximum length in the batch
    data_padded = pad_sequence(data, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)


    # TODO - pad like so for each size of input data & lables
    
    #  Calculate how many zero vectors are needed to padd to MAX_TANSOR_LENGTH
    data_padding_num = max_tansor_length - data_padded.shape[1]
    lables_padding_num = max_tansor_length - labels_padded.shape[1]

    # Check if we need to add any vectors
    if data_padding_num > 0:
        # Create a tensor of zero vectors with the same batch size and vector size, on the same device
        zero_vector_data = torch.zeros(data_padded.shape[0], data_padding_num, data_padded.shape[2], device=data_padded.device)
        
        # Concatenate the original tensor with the zero vectors along dimension 1
        data_padded = torch.cat([data_padded, zero_vector_data], dim=1)

    if lables_padding_num > 0:
        # Create a tensor of zero vectors with the same batch size and vector size, on the same device
        zero_vector_labels = torch.zeros(labels_padded.shape[0], lables_padding_num, labels_padded.shape[2], device=labels_padded.device)
        
        # Concatenate the original tensor with the zero vectors along dimension 1
        labels_padded = torch.cat([labels_padded, zero_vector_labels], dim=1)        
        
    
    # Create masks for data and labels
    data_masks = (data_padded != 0).any(dim=-1).float()
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
        
    # Load dataset
    loaded_data = torch.load(dataset_path)
    
    data = [pad(details[0]) for _, details in loaded_data.items()]
    labels = [pad(details[1]) for _, details in loaded_data.items()]
    
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


def train_model(model, criterion, optimizer, dataset_path: str, epochs=EPOCHS, batch_size = BATCH_SIZE) -> tuple:
    
    # Adjust DataLoader batch size
    train_loader, val_loader, test_loader = create_data_loaders(dataset_path, batch_size)
    
    print("Data Loaders created!")
    
    # Train the model
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        train_loss = 0
        for data, labels, data_masks, labels_masks in train_loader:

            optimizer.zero_grad()  # Zero out any gradients from previous steps
            # output = model(data[:,:2,:], src_key_padding_mask=data_masks[:,:2])  # Ensure masks are used
            output = model(data, src_key_padding_mask=data_masks)  # Ensure masks are used
            loss = criterion(output, labels)  # Calculate loss
            loss.backward(retain_graph=True)  # Backpropagate the error
            optimizer.step()  # Update model parameters
            train_loss += loss.item()
        train_loss /= len(train_loader)  # Average the loss over the batch

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        validation_loss = 0
        for data, labels, data_masks, labels_masks in val_loader:
            # print("Data shape:", data.shape)  # Should be [batch_size, seq_length, feature_size]
            # print("Lables shape:", labels.shape)
            # print("Data masks shape:", data_masks.shape)  # Should match data's seq_length
            with torch.no_grad():  # No gradient calculation
                # output = model(data[:,:2,:], src_key_padding_mask=data_masks[:,:2])  # Use masks during validation as well
                output = model(data, src_key_padding_mask=data_masks)  # Use masks during validation as well
                validation_loss += criterion(output, labels).item()  # Accumulate validation loss
        validation_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}, Validation Loss: {validation_loss:.4f}")
    
    return model, test_loader


def test_model(model, test_loader, criterion):
    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        validation_loss = sum(criterion(model(data), targets) for data, targets in test_loader) / len(test_loader)
    print(f"Test Loss: {validation_loss}")




def find_best_hypers(trial, dataset_path: str):

    # Define the hyperparameters to tune
    lr = trial.suggest_float('lr', 1e-6, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    num_layers = trial.suggest_categorical('num_layers', [1, 2])
    num_heads = trial.suggest_categorical('num_heads', [1, 2])
    dim_feedforward = trial.suggest_categorical('dim_feedforward', [16, 32, 64, 128, 256])
    dropout = trial.suggest_categorical('dropout', [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])


    # Create the model, criterion, and optimizer
    model = HiddenStateTransformer(num_layers=num_layers, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Adjust DataLoader batch size
    train_loader, val_loader, test_loader = create_data_loaders(dataset_path, batch_size=batch_size)
    
    print("Data Loaders created!")
    
    # Train the model
    for epoch in range(EPOCHS):
        model.train()  # Set the model to training mode
        train_loss = 0
        for data, labels, data_masks, labels_masks in train_loader:
            # print("Data shape:", data.shape)  # Should be [batch_size, seq_length, feature_size]
            # print("Lables shape:", labels.shape)
            # print("Data masks shape:", data_masks.shape)  # Should match data's seq_length

            optimizer.zero_grad()  # Zero out any gradients from previous steps
            # output = model(data[:,:2,:], src_key_padding_mask=data_masks[:,:2])  # Ensure masks are used
            output = model(data, src_key_padding_mask=data_masks)
            loss = criterion(output, labels)  # Calculate loss
            loss.backward(retain_graph=True)  # Backpropagate the error
            optimizer.step()  # Update model parameters
            train_loss += loss.item()
        train_loss /= len(train_loader)  # Average the loss over the batch

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        validation_loss = 0
        for data, labels, data_masks, labels_masks in val_loader:

            with torch.no_grad():  # No gradient calculation
                # output = model(data[:,:2,:], src_key_padding_mask=data_masks[:,:2])  # Use masks during validation as well
                output = model(data, src_key_padding_mask=data_masks)  # Use masks during validation as well
                validation_loss += criterion(output, labels).item()  # Accumulate validation loss
        validation_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}, Validation Loss: {validation_loss:.4f}")

        # Report intermediate objective value.
        trial.report(validation_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            print("/n/n==========its because of me!!==========/n/n", flush=True)
            raise optuna.exceptions.TrialPruned()
        # print("/n/n==========its not because of me!!==========/n/n", flush=True)

    return validation_loss
