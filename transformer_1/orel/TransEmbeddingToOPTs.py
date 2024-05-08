import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import torch.optim as optim


MAX_TANSOR_LENGTH = 1024


# Transformer
class HiddenStateTransformer(nn.Module):
    def __init__(self, input_size, output_size, num_layers, num_heads, dim_feedforward, dropout=0.1):
        super(HiddenStateTransformer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Transformer Encoder Layer
        encoder_layers = TransformerEncoderLayer(d_model=input_size, nhead=num_heads,
                                                 dim_feedforward=dim_feedforward, dropout=dropout, activation=F.relu)
        encoder_layers.self_attn.batch_first = True
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers,
                                                      enable_nested_tensor=1 - (num_heads % 2))

        # Linear layer to map to the target hidden size
        self.fc = nn.Linear(input_size, 512)

    def forward(self, src):
        # src shape: (seq_length, batch_size, input_size)
        encoded = self.transformer_encoder(src)
        # encoded shape: (seq_length, batch_size, input_size)
        output = self.fc(encoded)
        # output shape: (seq_length, batch_size, output_size)
        return output


def pad_tensor_to_length(tensor, target_length: int = MAX_TANSOR_LENGTH):
    # Get the current length of the tensor
    current_length = tensor.size(0)
    
    # Calculate how much padding is needed
    if current_length < target_length:
        # Pad the tensor if it is shorter than the target length
        padding_size = target_length - current_length
        padding = torch.zeros((padding_size,) + tensor.size()[1:])
        padded_tensor = torch.cat([tensor, padding], dim=0)
    else:
        # Truncate the tensor if it is longer than the target length
        padded_tensor = tensor[:target_length]
    
    return padded_tensor



def create_data_loders(dataset_path: str) -> tuple:
    
    # TODO - Padd with zeros. <--------------------------
    
    # Load dataset
    loaded_data = torch.load(dataset_path)
    
    data = [details[1] for _, details in loaded_data.items()]
    labels = [details[2] for _, details in loaded_data.items()]

    # First, split into training and temp
    data_train, data_temp, labels_train, labels_temp = train_test_split(data, labels, test_size=0.25, random_state=42)  # 75% train, 25% temp

    # Now split temp into validation and test
    data_val, data_test, labels_val, labels_test = train_test_split(data_temp, labels_temp, test_size=0.6, random_state=42)  # 40% of temp to validation, 60% of temp to test

    # Convert data to PyTorch tensors and wrap them in a dataset
    train_dataset = TensorDataset(data_train, labels_train)
    val_dataset = TensorDataset(data_val, labels_val)
    test_dataset = TensorDataset(data_test, labels_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train(model, criterion, optimizer, dataset_path: str, epochs: int = 10) -> tuple:
    
    train_loader,val_loader, test_loader = create_data_loders(dataset_path)
    
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        for data, targets in train_loader:
            optimizer.zero_grad()
            predicted_embeddings = model(data)
            loss = criterion(predicted_embeddings, targets)
            loss.backward()
            optimizer.step()
        
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            validation_loss = sum(criterion(model(data), targets) for data, targets in val_loader) / len(val_loader)

        print(f"Epoch {epoch+1}, Loss: {loss.item()}, Validation Loss: {validation_loss}")
    
    return model, test_loader



model = HiddenStateTransformer()  # Your defined nn.Module

criterion = nn.MSELoss()  # Example loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)

train(model, criterion, optimizer,"resources/big_one_token_dataset.pt")