
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class BlendDataset(Dataset):
    def __init__(self, blend_data, smiles_map, descriptors_map):
        self.blend_data = blend_data
        self.smiles_map = smiles_map
        self.descriptors_map = descriptors_map

    def __len__(self):
        return len(self.blend_data)

    def __getitem__(self, idx):
        blend_row = self.blend_data.iloc[idx]

        # Ensure MON and RON values are floats and handle NaNs
        # In `data_preprocessing.py`, NaNs in MON/RON will be handled before dataset creation.
        # For now, `np.nan_to_num` is kept for robustness, assuming inputs might still contain NaNs.
        mon_ron_values = blend_row[['MON', 'RON','CN']].values.astype(np.float32)
        mon_ron_values = np.nan_to_num(mon_ron_values, nan=0.0)
        target_mon_ron = torch.FloatTensor(mon_ron_values)

        # Extract component SMILES and molar fractions for the current blend
        component_smiles = []
        molar_fractions = []
        # Iterate over all columns to find SMILES keys present in smiles_map
        for smiles_col in blend_row.index:
            if smiles_col in self.smiles_map and not pd.isna(blend_row[smiles_col]):
                component_smiles.append(smiles_col)
                molar_fractions.append(blend_row[smiles_col])

        # Retrieve pre-processed smiles sparse vectors and scaled descriptors
        # Handle cases where component_smiles might be empty (e.g., if a blend has no valid components after filtering)
        if not component_smiles:
            # Return dummy tensors or raise an error depending on desired behavior
            # For now, return empty tensors and let collate_fn handle it or fail gracefully
            smiles_inputs = torch.empty(0)
            descriptor_inputs = torch.empty(0)
            molar_fractions_tensor = torch.empty(0)
        else:
            smiles_inputs = torch.stack([self.smiles_map[s] for s in component_smiles])
            descriptor_inputs = torch.stack([self.descriptors_map[s] for s in component_smiles])
            molar_fractions_tensor = torch.FloatTensor(molar_fractions).unsqueeze(1)

        return smiles_inputs, descriptor_inputs, molar_fractions_tensor, target_mon_ron

def collate_blend_batch(batch):
    # batch is a list of tuples: (smiles_inputs, descriptor_inputs, molar_fractions, target_mon_ron)

    all_smiles_inputs = []
    all_descriptor_inputs = []
    all_molar_fractions = []
    blend_indices = [] # To keep track of which molecule belongs to which blend
    blend_targets = []

    molecule_count = 0
    for i, (smiles_i, descriptors_i, molar_fractions_i, targets_i) in enumerate(batch):
        num_molecules_in_blend = smiles_i.size(0)

        if num_molecules_in_blend > 0: # Only process if there are actual molecules in the blend
            all_smiles_inputs.append(smiles_i)
            all_descriptor_inputs.append(descriptors_i)
            all_molar_fractions.append(molar_fractions_i)
            blend_indices.extend([i] * num_molecules_in_blend)
            molecule_count += num_molecules_in_blend

        blend_targets.append(targets_i)

    # Handle cases where all blends in a batch might be empty or resulting tensors are empty
    if not all_smiles_inputs:
        # Return empty tensors of appropriate shape or handle as error
        # For now, returning empty tensors. The model might need to handle these gracefully.
        return (
            torch.empty(0, 0, 0), # smiles_inputs
            torch.empty(0, 0),    # descriptor_inputs
            torch.empty(0, 1),    # molar_fractions
            torch.LongTensor([]), # blend_indices
            torch.stack(blend_targets) if blend_targets else torch.empty(0, 2) # targets
        )

    return (
        torch.cat(all_smiles_inputs, dim=0),
        torch.cat(all_descriptor_inputs, dim=0),
        torch.cat(all_molar_fractions, dim=0),
        torch.LongTensor(blend_indices),
        torch.stack(blend_targets)
    )
