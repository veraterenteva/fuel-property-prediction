import blending_optimizer
import torch
import config
import pandas as pd
import pathlib

from data_preprocessing import get_smiles_vocabulary, get_top_n_descriptors, preprocess_data
from models import CombinedModel

project_root = str(pathlib.Path.cwd())

df_pure = pd.read_csv(project_root + '/data/pure_for_mix.csv')

df_mix = pd.read_csv(project_root + '/data/mix_combined_cn.csv')

not_smiles = ('index','Mixture name','Dataset','RON','MON','CN')
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

print("\n--- K-Component Fuel Blend Optimization ---")

targets_provided = 0
target_ron = None
target_mon = None
target_cn = None

while targets_provided == 0 or targets_provided > 2:
    try:
        ron_input = input("Enter desired RON (leave blank for optional): ")
        target_ron = float(ron_input) if ron_input.strip() != '' else None

        mon_input = input("Enter desired MON (leave blank for optional): ")
        target_mon = float(mon_input) if mon_input.strip() != '' else None

        cn_input = input("Enter desired CN (leave blank for optional): ")
        target_cn = float(cn_input) if cn_input.strip() != '' else None

        targets_provided = sum([1 for x in [target_ron, target_mon, target_cn] if x is not None])

        if targets_provided == 0:
            print("At least one target (RON, MON, or CN) must be provided. Please try again.")
        elif targets_provided > 2:
            print("You can only leave up to two targets blank (i.e., at least one must be provided, and not all three). Please try again.")

    except ValueError:
        print("Invalid input for RON, MON, or CN. Please enter numeric values or leave blank. Please try again.")
        targets_provided = 0 # Reset to force re-entry

try:
    k_components = int(input("Enter the desired number of components (k): "))
    num_trials = int(input("Enter the number of trials for stochastic search (e.g., 100-1000 for good exploration): "))
except ValueError:
    print("Invalid input for k or trials. Using default values.")
    k_components = 3 # Default to 3 components
    num_trials = 100 # Default trials

print(f"Using values: Target RON={target_ron}, MON={target_mon}, CN={target_cn}, k={k_components}, trials={num_trials}")


optimal_comp_k, pred_ron_k, pred_mon_k, pred_cn_k, final_loss_k = blending_optimizer.find_k_component_blend(
    target_ron=target_ron,
    target_mon=target_mon,
    target_cn=target_cn,
    k_components=k_components,
    all_available_smiles=all_smiles_in_mix,
    smiles_map=smiles_map,              
    descriptors_map=descriptors_map,   
    model=model_for_optimizer,
    device=device,
    num_trials=num_trials
)

if optimal_comp_k:
    print(f"\nOptimization successful for requested targets with {len(optimal_comp_k)} components.")
    if target_ron is not None: print(f"Target RON: {target_ron}, Predicted RON: {pred_ron_k:.2f}")
    if target_mon is not None: print(f"Target MON: {target_mon}, Predicted MON: {pred_mon_k:.2f}")
    if target_cn is not None: print(f"Target CN: {target_cn}, Predicted CN: {pred_cn_k:.2f}")
    print(f"Final Objective Function Value (Squared Error): {final_loss_k:.4f}")
    print("Optimal K-Component Blend Composition:")
    for smiles, fraction in sorted(optimal_comp_k.items(), key=lambda item: item[1], reverse=True):
        print(f"  {smiles}: {fraction:.4f}")
else:
    print("Could not find an optimal k-component blend.")
