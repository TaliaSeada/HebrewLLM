import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# import sys
# from pathlib import Path # if you haven't already done so
# file = Path(__file__).resolve()
# parent, root = file.parent, file.parents[1]
# sys.path.append(str(root))

from model.config import MAX_TANSOR_LENGTH, DROPOUT, EPOCHS, BATCH_SIZE
from model.absTransformer import AbstractHiddenStateTransformer
from data.dataManipulation import create_data_loaders


# Transformer
class HiddenStateTransformer(AbstractHiddenStateTransformer):
    def __init__(self, input_size=1024, output_size=MAX_TANSOR_LENGTH, num_layers=2, num_heads=1, dim_feedforward=128, dropout=DROPOUT, activation=F.relu):
        super(HiddenStateTransformer, self).__init__(input_size, output_size, num_layers, num_heads, dim_feedforward, dropout, activation)
        self.transformer_encoder = self.build_transformer_encoder()

    def build_transformer_encoder(self):
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.num_heads,
                                                    dim_feedforward=self.dim_feedforward, dropout=self.dropout, activation=self.activation)
        encoder_layers.self_attn.batch_first = True
        transformer_encoder = nn.TransformerEncoder(encoder_layers, self.num_layers)
        return transformer_encoder

    def forward(self, src, src_key_padding_mask=None):
        encoded = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        return encoded


class HiddenStateTransformer2(nn.Module):
    def __init__(self, input_size, output_size, num_layers, num_heads, dim_feedforward, dropout=0.1):
        super(HiddenStateTransformer2, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Transformer Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads,
                                                 dim_feedforward=dim_feedforward, dropout=dropout, activation=F.relu)
        encoder_layers.self_attn.batch_first = True
        # encoder_layers.activation_relu_or_gelu=True
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers,
                                                      enable_nested_tensor=1 - (num_heads % 2))

        # Linear layer to map to the target hidden size
        self.fc = nn.Linear(input_size, 512)

    def forward(self, src):
        # src shape: (seq_length, batch_size, input_size)
        encoded = self.transformer_encoder(src)
        # encoded shape: (seq_length, batch_size, input_size)
        # Apply ReLU activation function
        # activated = F.relu(encoded)
        output = self.fc(encoded)
        # output shape: (seq_length, batch_size, output_size)
        return output
    

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
    
    return model, test_loader, train_loader, val_loader


def test_model(model, test_loader, criterion):
    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        validation_loss = sum(criterion(model(data), targets) for data, targets in test_loader) / len(test_loader)
    print(f"Test Loss: {validation_loss}")