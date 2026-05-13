
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Import config constants
import config

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    n_epochs=config.N_EPOCHS,
    l2_reg_lambda=config.L2_REG_LAMBDA,
    l1_reg_lambda=config.L1_REG_LAMBDA,
    patience=config.PATIENCE
):
    """
    Trains a PyTorch model with early stopping.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        criterion (nn.Module): The loss function.
        device (torch.device): The device (cpu or cuda) to run the training on.
        n_epochs (int): Maximum number of training epochs.
        l2_reg_lambda (float): Lambda for L2 regularization.
        l1_reg_lambda (float): Lambda for L1 regularization.
        patience (int): Number of epochs to wait for improvement before early stopping.

    Returns:
        tuple: (model, best_val_loss)
               The trained model and the best validation loss achieved.
    """
    model.to(device)
    print(f"Model moved to {device}.")

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for all_smiles_inputs, all_descriptor_inputs, all_molar_fractions, blend_indices, blend_targets in train_loader:
            all_smiles_inputs = all_smiles_inputs.to(device)
            all_descriptor_inputs = all_descriptor_inputs.to(device)
            all_molar_fractions = all_molar_fractions.to(device)
            blend_indices = blend_indices.to(device)
            blend_targets = blend_targets.to(device)

            optimizer.zero_grad()
            predictions = model(
                all_smiles_inputs,
                all_descriptor_inputs,
                all_molar_fractions,
                blend_indices
            )

            # Create a mask to ignore targets that were originally NaN and filled with 0.0
            # This assumes that targets are set to 0.0 where they are NaN in the original data
            mask_targets = (blend_targets != 0.0)

            batch_loss_unreduced = criterion(predictions, blend_targets)
            masked_loss = batch_loss_unreduced[mask_targets]

            if masked_loss.numel() > 0:
                loss = masked_loss.mean()
            else:
                # If all targets in the batch are masked out, the loss for this batch is 0.
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            loss_print = loss.item()

            # L2 regularization
            l2_norm = sum((p**2).sum() for p in model.parameters())
            loss = loss + l2_reg_lambda * l2_norm

            # L1 regularization
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_reg_lambda * l1_norm

            loss.backward()
            optimizer.step()
            train_loss += loss_print

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for all_smiles_inputs, all_descriptor_inputs, all_molar_fractions, blend_indices, blend_targets in val_loader:
                all_smiles_inputs = all_smiles_inputs.to(device)
                all_descriptor_inputs = all_descriptor_inputs.to(device)
                all_molar_fractions = all_molar_fractions.to(device)
                blend_indices = blend_indices.to(device)
                blend_targets = blend_targets.to(device)

                predictions = model(
                    all_smiles_inputs,
                    all_descriptor_inputs,
                    all_molar_fractions,
                    blend_indices
                )

                mask_targets = (blend_targets != 0.0)

                batch_loss_unreduced = criterion(predictions, blend_targets)
                masked_loss = batch_loss_unreduced[mask_targets]

                if masked_loss.numel() > 0:
                    loss = masked_loss.mean()
                else:
                    loss = torch.tensor(0.0, device=device)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping! No improvement in validation loss for {patience} epochs.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Loaded best model weights.")

    print("\nTraining complete.")
    return model, best_val_loss
