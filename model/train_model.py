import pandas as pd
import torch
from torch.utils.data import DataLoader

import pathlib

import config
import models
import datasets
import data_preprocessing
import training
import evaluation

import random
import numpy as np

np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
random.seed(config.SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

project_root = str(pathlib.Path.cwd())

df_pure = pd.read_csv(project_root + '/data/pure_for_mix.csv')

df_mix = pd.read_csv(project_root + '/data/mix_combined.csv')

not_smiles = ('index','Mixture name','Dataset','RON','MON')
all_smiles_in_mix = list(filter(lambda x: x not in not_smiles, df_mix.columns))

print(f"Loaded {len(df_pure)} pure components and {len(df_mix)} mixture entries.")
print(f"Found {len(all_smiles_in_mix)} unique SMILES in mixtures.")

char_to_idx, idx_to_char, vocab_size, max_seq_len = data_preprocessing.get_smiles_vocabulary(all_smiles_in_mix)

print(f"SMILES vocabulary size: {vocab_size}")
print(f"Max SMILES sequence length: {max_seq_len}")

selected_descriptor_names, df_pure_rdkit = data_preprocessing.get_top_n_descriptors(
    df_pure_components=df_pure,
    top_n_descriptors=config.TOP_N_DESCRIPTORS,
    seed=config.SEED
)

print(f"Selected {len(selected_descriptor_names)} top descriptors.")

# Preprocess all data (SMILES to sparse vectors, RDKit descriptors scaling, data splitting)
smiles_map, descriptors_map, train_blend_data, val_blend_data, scaler = data_preprocessing.preprocess_data(
    df_mix=df_mix,
    df_pure_components=df_pure,
    all_smiles_in_mix=all_smiles_in_mix,
    selected_descriptor_names=selected_descriptor_names,
    char_to_idx=char_to_idx,
    max_seq_len=max_seq_len,
    split_random=config.SPLIT_RANDOM,
    seed=config.SEED
)

print(f"Preprocessed data for {len(smiles_map)} unique SMILES.")
print(f"Training blends: {len(train_blend_data)}, Validation blends: {len(val_blend_data)}")


# Create BlendDataset instances
train_dataset = datasets.BlendDataset(train_blend_data, smiles_map, descriptors_map)
val_dataset = datasets.BlendDataset(val_blend_data, smiles_map, descriptors_map)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(config.SEED)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
    collate_fn=datasets.collate_blend_batch, worker_init_fn=seed_worker, generator=g
)
val_loader = DataLoader(
    val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
    collate_fn=datasets.collate_blend_batch
)

print(f"Train DataLoader created with {len(train_loader)} batches.")
print(f"Validation DataLoader created with {len(val_loader)} batches.")


model = models.CombinedModel(
    smiles_vocab_size=vocab_size,
    smiles_embedding_dim=config.SMILES_EMBEDDING_DIM,
    smiles_hidden_dims=config.SMILES_HIDDEN_DIMS,
    smiles_dropout=config.SMILES_DROPOUT,
    smiles_linear_output_dim=config.SMILES_LINEAR_OUTPUT_DIM,
    descriptor_input_dim=len(selected_descriptor_names),
    descriptor_hidden_dim1=config.DESCRIPTOR_HIDDEN_DIM1,
    descriptor_hidden_dim2=config.DESCRIPTOR_HIDDEN_DIM2,
    descriptor_output_dim=config.DESCRIPTOR_OUTPUT_DIM,
    predictor_hidden_dim1=config.PREDICTOR_HIDDEN_DIM1,
    predictor_hidden_dim2=config.PREDICTOR_HIDDEN_DIM2,
    predictor_hidden_dim3=config.PREDICTOR_HIDDEN_DIM3,
    output_dim=config.OUTPUT_DIM
)

optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
criterion = torch.nn.L1Loss(reduction='none') # Use reduction='none' to apply mask

print("Model, optimizer, and criterion initialized.")

trained_model, best_val_loss = training.train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    n_epochs=config.N_EPOCHS,
    l2_reg_lambda=config.L2_REG_LAMBDA,
    l1_reg_lambda=config.L1_REG_LAMBDA,
    patience=config.PATIENCE
)

# Save the best model state dictionary
model_save_path = 'best_model.pth'
torch.save(trained_model.state_dict(), model_save_path)
print(f"Best model state saved to {model_save_path}")

evaluation.evaluate_model(
    model=trained_model,
    data_loader=val_loader,
    device=device,
    split_random=config.SPLIT_RANDOM,
    seed=config.SEED,
    l2_reg_lambda=config.L2_REG_LAMBDA,
    smiles_hidden_dims=config.SMILES_HIDDEN_DIMS,
    smiles_dropout=config.SMILES_DROPOUT,
    smiles_linear_output_dim=config.SMILES_LINEAR_OUTPUT_DIM,
    top_n_descriptors=config.TOP_N_DESCRIPTORS,
    descriptor_input_dim=len(selected_descriptor_names),
    descriptor_hidden_dim1=config.DESCRIPTOR_HIDDEN_DIM1,
    descriptor_hidden_dim2=config.DESCRIPTOR_HIDDEN_DIM2,
    descriptor_output_dim=config.DESCRIPTOR_OUTPUT_DIM,
    predictor_hidden_dim1=config.PREDICTOR_HIDDEN_DIM1,
    predictor_hidden_dim2=config.PREDICTOR_HIDDEN_DIM2,
    predictor_hidden_dim3=config.PREDICTOR_HIDDEN_DIM3,
    output_dim=config.OUTPUT_DIM,
    output_json_path='evaluation_results.json',
    plots_path='predicted_vs_actual_plots.png'
)
