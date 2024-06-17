import optuna
import torch
from model.HiddenStateTransformer import HiddenStateTransformer
from model.config import EPOCHS

def find_best_hypers(model, criterion, optimizer, trial, dataset_path: str):
    from data.dataManipulation import create_data_loaders

    # Define the hyperparameters to tune
    lr = trial.suggest_float('lr', 1e-6, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    num_layers = trial.suggest_categorical('num_layers', [1])
    num_heads = trial.suggest_categorical('num_heads', [1, 2])
    dim_feedforward = trial.suggest_categorical('dim_feedforward', [16, 32, 64, 128, 256])
    dropout = trial.suggest_categorical('dropout', [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])


    # Create the model, criterion, and optimizer
    model = HiddenStateTransformer(num_layers=num_layers, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)


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


def findBest(model, dataset_path):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: find_best_hypers(model, trial, dataset_path), n_trials=100)

    # Print best hyperparameters
    print(study.best_params)
