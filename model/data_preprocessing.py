
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Import config constants
import config

def get_smiles_vocabulary(smiles_list):
    """
    Builds character vocabulary and determines max sequence length from a list of SMILES.
    """
    all_chars = sorted(list(set(''.join(smiles_list))))
    char_to_idx = {char: i + 1 for i, char in enumerate(all_chars)} # +1 for padding
    idx_to_char = {i + 1: char for i, char in enumerate(all_chars)}
    char_to_idx['<PAD>'] = 0
    idx_to_char[0] = '<PAD>'

    vocab_size = len(char_to_idx)
    max_seq_len = max(len(s) for s in smiles_list)

    return char_to_idx, idx_to_char, vocab_size, max_seq_len

def smiles_to_sparse_vectors(smiles_list, char_to_idx, max_seq_len):
    """
    Converts a list of SMILES strings into a list of sparse (one-hot) vector matrices.
    Each matrix has dimensions (vocab_size, max_seq_len).
    A '1' at [char_idx, pos] indicates the character at char_idx is present at pos.
    Each SMILES string is padded/truncated to max_seq_len.
    """
    sparse_vectors_list = []
    vocab_size = len(char_to_idx)

    for smiles in smiles_list:
        smiles_matrix = torch.zeros(vocab_size, max_seq_len)
        for i, char in enumerate(smiles[:max_seq_len]):
            idx = char_to_idx.get(char, char_to_idx['<PAD>'])
            smiles_matrix[idx, i] = 1
        sparse_vectors_list.append(smiles_matrix)
    return sparse_vectors_list

def calculate_rdkit_descriptors(smiles):
    """
    Calculates RDKit molecular descriptors for a given SMILES string.
    Returns a dictionary of descriptor names and their values.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    descriptor_names = [d[0] for d in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    descriptors = calculator.CalcDescriptors(mol)
    return dict(zip(descriptor_names, descriptors))

def get_top_n_descriptors(df_pure_components, top_n_descriptors=config.TOP_N_DESCRIPTORS, seed=config.SEED):
    """
    Calculates RDKit descriptors for pure components, uses RandomForest to select top N.
    Args:
        df_pure_components (pd.DataFrame): DataFrame of pure components with 'SMILES', 'MON', 'RON'.
        top_n_descriptors (int): Number of top descriptors to select.
        seed (int): Random seed for reproducibility.
    Returns:
        list: List of names of the top N selected descriptors.
        pd.DataFrame: DataFrame with RDKit descriptors for all pure components (before filtering for top N).
    """
    df_pure_rdkit_raw = df_pure_components['SMILES'].apply(calculate_rdkit_descriptors)
    df_pure_rdkit = pd.DataFrame(df_pure_rdkit_raw.tolist(), index=df_pure_rdkit_raw.index)

    # Combine RDKit descriptors with targets for feature importance calculation
    df_pure_with_descriptors = pd.concat([df_pure_components, df_pure_rdkit], axis=1)

    # Prepare data for Random Forest feature selection
    # Drop rows with NaNs in targets for RF (as it doesn't handle them)
    X = df_pure_with_descriptors.drop(columns=['Dataset','SMILES', 'MON','RON'], errors='ignore')
    y = df_pure_with_descriptors[['MON','RON']]

    combined_data_for_rf = pd.concat([X, y], axis=1)
    combined_data_clean_for_rf = combined_data_for_rf.dropna()

    X_rf = combined_data_clean_for_rf[X.columns]
    y_rf = combined_data_clean_for_rf[['MON','RON']]

    if X_rf.empty:
        print("Warning: No complete data rows for RDKit descriptor selection. Returning all descriptors.")
        return X.columns.tolist(), df_pure_rdkit

    X_train, _, y_train, _ = train_test_split(X_rf, y_rf, test_size=0.2, random_state=seed) # Use only training part for RF

    rf_model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    feature_importances = pd.Series(rf_model.feature_importances_, index=X_rf.columns)
    sorted_importances = feature_importances.sort_values(ascending=False)
    selected_descriptor_names = sorted_importances.head(top_n_descriptors).index.tolist()

    return selected_descriptor_names, df_pure_rdkit

def preprocess_data(df_mix, df_pure_components, all_smiles_in_mix, selected_descriptor_names, char_to_idx, max_seq_len, scaler=None, split_random=False, seed=config.SEED):
    """
    Orchestrates the preprocessing steps to prepare data for the model.

    Args:
        df_mix (pd.DataFrame): DataFrame containing mixture data.
        df_pure_components (pd.DataFrame): DataFrame of pure components (used for descriptor scaling).
        all_smiles_in_mix (list): List of all unique SMILES strings found in the mixture dataset.
        selected_descriptor_names (list): List of RDKit descriptor names selected for use.
        char_to_idx (dict): Mapping from character to index for SMILES tokenization.
        max_seq_len (int): Maximum sequence length for SMILES.
        scaler (StandardScaler, optional): Pre-fitted scaler. If None, a new one is fitted.
        split_random (bool): If True, splits data randomly; otherwise uses 'Dataset' column.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (smiles_map, descriptors_map, train_blend_data, val_blend_data, scaler)
               smiles_map (dict): Maps SMILES strings to their sparse vector representations.
               descriptors_map (dict): Maps SMILES strings to their scaled descriptor vectors.
               train_blend_data (pd.DataFrame): Training blend data.
               val_blend_data (pd.DataFrame): Validation blend data.
               scaler (StandardScaler): Fitted StandardScaler.
    """
    # Calculate RDKit descriptors for unique SMILES present in the mixtures
    df_unique_smiles_for_mix_desc_raw = pd.DataFrame({'SMILES': all_smiles_in_mix})
    df_unique_smiles_for_mix_desc = df_unique_smiles_for_mix_desc_raw['SMILES'].apply(calculate_rdkit_descriptors)
    df_unique_smiles_rdkit = pd.DataFrame(df_unique_smiles_for_mix_desc.tolist(), index=df_unique_smiles_for_mix_desc.index)

    # Use the full set of descriptors from pure components to fit the scaler
    # This ensures consistency in scaling, as pure_for_mix.csv contains the full range.
    df_all_descriptors_for_scaling_raw = df_pure_components['SMILES'].apply(calculate_rdkit_descriptors)
    df_all_descriptors_for_scaling = pd.DataFrame(df_all_descriptors_for_scaling_raw.tolist())

    descriptor_values_to_fit = df_all_descriptors_for_scaling[selected_descriptor_names].values

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(descriptor_values_to_fit) # Fit scaler on all available descriptors from pure components

    # Transform descriptors for all unique SMILES in the mixtures
    scaled_descriptor_values_for_mix = scaler.transform(df_unique_smiles_rdkit[selected_descriptor_names].values)

    smiles_map = {}
    descriptors_map = {}
    for i, smiles_string in enumerate(all_smiles_in_mix):
        smiles_map[smiles_string] = smiles_to_sparse_vectors([smiles_string], char_to_idx, max_seq_len)[0]
        descriptors_map[smiles_string] = torch.FloatTensor(scaled_descriptor_values_for_mix[i])

    # Ensure 'MON' and 'RON' columns are numeric and handle NaNs in blend targets
    # Fill NaN values with 0.0, and the training loop will mask these out.
    df_mix['MON'] = pd.to_numeric(df_mix['MON'], errors='coerce').fillna(0.0)
    df_mix['RON'] = pd.to_numeric(df_mix['RON'], errors='coerce').fillna(0.0)

    if split_random:
        all_blend_indices = np.arange(len(df_mix))
        train_blend_idx, val_blend_idx = train_test_split(all_blend_indices, test_size=0.2, random_state=seed)

        train_blend_data = df_mix.iloc[train_blend_idx].copy()
        val_blend_data = df_mix.iloc[val_blend_idx].copy()
    else:
        train_blend_data = df_mix[df_mix['Dataset'] == 'train'].copy()
        val_blend_data = df_mix[df_mix['Dataset'] == 'test'].copy()

    return smiles_map, descriptors_map, train_blend_data, val_blend_data, scaler
