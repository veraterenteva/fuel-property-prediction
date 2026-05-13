import blending_optimizer
import torch
import config
import pandas as pd
import pathlib

from data_preprocessing import get_smiles_vocabulary, get_top_n_descriptors, preprocess_data
from models import CombinedModel

project_root = str(pathlib.Path.cwd())

df_pure = pd.read_csv(project_root + '/data/pure_for_mix.csv')

df_mix = pd.read_csv(project_root + '/data/mix_combined.csv')

not_smiles = ('index','Mixture name','Dataset','RON','MON')
all_smiles_in_mix = list(filter(lambda x: x not in not_smiles, df_mix.columns))

print(f"Loaded {len(df_pure)} pure components and {len(df_mix)} mixture entries.")
print(f"Found {len(all_smiles_in_mix)} unique SMILES in mixtures.")

char_to_idx, idx_to_char, vocab_size, max_seq_len = get_smiles_vocabulary(all_smiles_in_mix)

selected_descriptor_names, df_pure_rdkit = get_top_n_descriptors(
    df_pure_components=df_pure,
    top_n_descriptors=config.TOP_N_DESCRIPTORS,
    seed=config.SEED
)

print(f"Selected {len(selected_descriptor_names)} top descriptors.")

# Preprocess all data (SMILES to sparse vectors, RDKit descriptors scaling, data splitting)
smiles_map, descriptors_map, train_blend_data, val_blend_data, scaler = preprocess_data(
    df_mix=df_mix,
    df_pure_components=df_pure,
    all_smiles_in_mix=all_smiles_in_mix,
    selected_descriptor_names=selected_descriptor_names,
    char_to_idx=char_to_idx,
    max_seq_len=max_seq_len,
    split_random=config.SPLIT_RANDOM,
    seed=config.SEED
)

# # Re-initialize the model architecture based on config and previously determined vocab_size and descriptor_input_dim
model_for_optimizer = CombinedModel(
     smiles_vocab_size=vocab_size,
     smiles_embedding_dim=config.SMILES_EMBEDDING_DIM,
     smiles_hidden_dims=config.SMILES_HIDDEN_DIMS,
     smiles_dropout=config.SMILES_DROPOUT,
     smiles_linear_output_dim=config.SMILES_LINEAR_OUTPUT_DIM,
     descriptor_input_dim=config.DESCRIPTOR_INPUT_DIM,
     descriptor_hidden_dim1=config.DESCRIPTOR_HIDDEN_DIM1,
     descriptor_hidden_dim2=config.DESCRIPTOR_HIDDEN_DIM2,
     descriptor_output_dim=config.DESCRIPTOR_OUTPUT_DIM,
     predictor_hidden_dim1=config.PREDICTOR_HIDDEN_DIM1,
     predictor_hidden_dim2=config.PREDICTOR_HIDDEN_DIM2,
     predictor_hidden_dim3=config.PREDICTOR_HIDDEN_DIM3,
     output_dim=config.OUTPUT_DIM
 )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_for_optimizer.load_state_dict(torch.load('best_model.pth',map_location=torch.device(device)))
model_for_optimizer.to(device)
model_for_optimizer.eval()


print("Model ready for optimization.")

# Define target properties for the blend
target_ron = float(input('ENTER DESIRED RON VALUE: '))
target_mon = float(input('ENTER DESIRED MON VALUE: '))
k_components = int(input('ENTER DESIRED NUMBER OF COMPONENTS: '))
num_trials = int(input('ENTER NUMBER OF DESIRED STOCHASTIC TRIALS: '))

print(f"\nAttempting to find a {k_components}-component blend for Target RON: {target_ron}, Target MON: {target_mon}")

optimal_comp_k, pred_ron_k, pred_mon_k, final_loss_k = blending_optimizer.find_k_component_blend(
    target_ron=target_ron,
    target_mon=target_mon,
    k_components=k_components,
    all_available_smiles=all_smiles_in_mix,
    smiles_map=smiles_map,              
    descriptors_map=descriptors_map,   
    model=model_for_optimizer,
    device=device,
    num_trials=num_trials
)

if optimal_comp_k:
    print(f"\nOptimization successful for Target RON: {target_ron}, MON: {target_mon} with {len(optimal_comp_k)} components.")
    print(f"Predicted RON: {pred_ron_k:.2f}, Predicted MON: {pred_mon_k:.2f}")
    print(f"Final Objective Function Value (Squared Error): {final_loss_k:.4f}")
    print("Optimal K-Component Blend Composition:")
    for smiles, fraction in sorted(optimal_comp_k.items(), key=lambda item: item[1], reverse=True):
        print(f"  {smiles}: {fraction:.4f}")
else:
    print(f"Could not find an optimal {k_components}-component blend for Target RON: {target_ron}, MON: {target_mon}.")
