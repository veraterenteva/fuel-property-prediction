
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import json

def evaluate_model(model, data_loader, device, split_random, seed, l2_reg_lambda, smiles_hidden_dims, smiles_dropout, smiles_linear_output_dim, top_n_descriptors, descriptor_input_dim, descriptor_hidden_dim1, descriptor_hidden_dim2, descriptor_output_dim, predictor_hidden_dim1, predictor_hidden_dim2, predictor_hidden_dim3, output_dim, output_json_path='evaluation_results.json', plots_path='predicted_vs_actual_plots.png'):
    """
    Evaluates the trained model on a given DataLoader and saves metrics and plots.

    Args:
        model (nn.Module): The trained PyTorch model.
        data_loader (DataLoader): DataLoader for the evaluation set.
        device (torch.device): The device (cpu or cuda) to run the evaluation on.
        split_random (bool): If True, data was split randomly; otherwise uses 'Dataset' column.
        seed (int): Random seed for reproducibility.
        l2_reg_lambda (float): L2 regularization lambda used during training.
        smiles_hidden_dims (list): SMILES encoder hidden dimensions.
        smiles_dropout (float): SMILES encoder dropout rate.
        smiles_linear_output_dim (int): SMILES encoder linear output dimension.
        top_n_descriptors (int): Number of top RDKit descriptors used.
        descriptor_input_dim (int): Descriptor encoder input dimension.
        descriptor_hidden_dim1 (int): Descriptor encoder first hidden dimension.
        descriptor_hidden_dim2 (int): Descriptor encoder second hidden dimension.
        descriptor_output_dim (int): Descriptor encoder output dimension.
        predictor_hidden_dim1 (int): Predictor first hidden dimension.
        predictor_hidden_dim2 (int): Predictor second hidden dimension.
        predictor_hidden_dim3 (int): Predictor third hidden dimension.
        output_dim (int): Predictor output dimension.
        output_json_path (str): Path to save evaluation metrics JSON.
        plots_path (str): Path to save predicted vs. actual plots.
    """
    model.eval()
    predictions_all_raw = []
    actuals_all_raw = []
    masks_all_raw = []

    with torch.no_grad():
        for all_smiles_inputs, all_descriptor_inputs, all_molar_fractions, blend_indices, blend_targets in data_loader:
            all_smiles_inputs = all_smiles_inputs.to(device)
            all_descriptor_inputs = all_descriptor_inputs.to(device)
            all_molar_fractions = all_molar_fractions.to(device)
            blend_indices = blend_indices.to(device)

            predictions = model(all_smiles_inputs, all_descriptor_inputs, all_molar_fractions, blend_indices)
            predictions_all_raw.append(predictions.cpu().numpy())

            mask_targets_batch = (blend_targets != 0.0).cpu().numpy()
            actuals_all_raw.append(blend_targets.cpu().numpy())
            masks_all_raw.append(mask_targets_batch)

    predictions_full = np.vstack(predictions_all_raw)
    actuals_full = np.vstack(actuals_all_raw)
    masks_full = np.vstack(masks_all_raw)

    # Filter predictions and actuals using masks for MON and RON separately
    mon_present_mask = masks_full[:, 0]
    actuals_mon = actuals_full[mon_present_mask, 0]
    predictions_mon = predictions_full[mon_present_mask, 0]

    ron_present_mask = masks_full[:, 1]
    actuals_ron = actuals_full[ron_present_mask, 1]
    predictions_ron = predictions_full[ron_present_mask, 1]

    cn_present_mask = masks_full[:, 2]
    actuals_cn = actuals_full[cn_present_mask, 2]
    predictions_cn = predictions_full[cn_present_mask, 2]

    # Calculate metrics for MON
    r2_mon = r2_score(actuals_mon, predictions_mon) if actuals_mon.shape[0] > 0 else np.nan
    mae_mon = mean_absolute_error(actuals_mon, predictions_mon) if actuals_mon.shape[0] > 0 else np.nan
    mape_mon = mean_absolute_percentage_error(actuals_mon, predictions_mon) * 100 if actuals_mon.shape[0] > 0 else np.nan

    # Calculate metrics for RON
    r2_ron = r2_score(actuals_ron, predictions_ron) if actuals_ron.shape[0] > 0 else np.nan
    mae_ron = mean_absolute_error(actuals_ron, predictions_ron) if actuals_ron.shape[0] > 0 else np.nan
    mape_ron = mean_absolute_percentage_error(actuals_ron, predictions_ron) * 100 if actuals_ron.shape[0] > 0 else np.nan

    # Calculate metrics for CN
    r2_cn = r2_score(actuals_cn, predictions_cn) if actuals_cn.shape[0] > 0 else np.nan
    mae_cn = mean_absolute_error(actuals_cn, predictions_cn) if actuals_cn.shape[0] > 0 else np.nan
    mape_cn = mean_absolute_percentage_error(actuals_cn, predictions_cn) * 100 if actuals_cn.shape[0] > 0 else np.nan

    # For overall metrics, concatenate the valid MON, RON, and CN values
    actuals_overall_valid = np.concatenate((actuals_mon, actuals_ron, actuals_cn)) if (actuals_mon.shape[0] > 0 or actuals_ron.shape[0] > 0 or actuals_cn.shape[0] > 0) else np.array([])
    predictions_overall_valid = np.concatenate((predictions_mon, predictions_ron, predictions_cn)) if (predictions_mon.shape[0] > 0 or predictions_ron.shape[0] > 0 or predictions_cn.shape[0] > 0) else np.array([])

    r2_overall = r2_score(actuals_overall_valid, predictions_overall_valid) if actuals_overall_valid.shape[0] > 0 else np.nan
    mae_overall = mean_absolute_error(actuals_overall_valid, predictions_overall_valid) if actuals_overall_valid.shape[0] > 0 else np.nan
    mape_overall = mean_absolute_percentage_error(actuals_overall_valid, predictions_overall_valid) * 100 if actuals_overall_valid.shape[0] > 0 else np.nan

    print(f"R2 score on the validation set for MON: {r2_mon:.4f}")
    print(f"R2 score on the validation set for RON: {r2_ron:.4f}")
    print(f"R2 score on the validation set for CN: {r2_cn:.4f}")
    print(f"Average R2 score on the validation set: {r2_overall:.4f}")

    print(f"\nMAE score on the validation set for MON: {mae_mon:.4f}")
    print(f"MAE score on the validation set for RON: {mae_ron:.4f}")
    print(f"MAE score on the validation set for CN: {mae_cn:.4f}")
    print(f"Average MAE score on the validation set: {mae_overall:.4f}")

    print(f"\nMAPE score on the validation set for MON (percents): {mape_mon:.2f}")
    print(f"MAPE score on the validation set for RON (percents): {mape_ron:.2f}")
    print(f"MAPE score on the validation set for CN (percents): {mape_cn:.2f}")
    print(f"Average MAPE score on the validation set: {mape_overall:.2f}")

    evaluation_metrics = {
        'R2_MON': float(r2_mon),
        'R2_RON': float(r2_ron),
        'R2_CN': float(r2_cn),
        'R2_Overall': float(r2_overall),
        'MAE_MON': float(mae_mon),
        'MAE_RON': float(mae_ron),
        'MAE_CN': float(mae_cn),
        'MAE_Overall': float(mae_overall),
        'MAPE_MON': float(mape_mon),
        'MAPE_RON': float(mape_ron),
        'MAPE_CN': float(mape_cn),
        'MAPE_Overall': float(mape_overall),
        'SPLIT_RANDOM': split_random,
        'SEED': seed,
        'L2_REG_LAMBDA' : l2_reg_lambda,
        'SMILES_HIDDEN_DIMS': smiles_hidden_dims,
        'SMILES_DROPOUT': smiles_dropout,
        'SMILES_LINEAR_OUTPUT_DIM': smiles_linear_output_dim,
        'TOP_N_DESCRIPTORS': top_n_descriptors,
        'DESCRIPTOR_INPUT_DIM': descriptor_input_dim,
        'DESCRIPTOR_HIDDEN_DIM1': descriptor_hidden_dim1,
        'DESCRIPTOR_HIDDEN_DIM2': descriptor_hidden_dim2,
        'DESCRIPTOR_OUTPUT_DIM': descriptor_output_dim,
        'PREDICTOR_HIDDEN_DIM1': predictor_hidden_dim1,
        'PREDICTOR_HIDDEN_DIM2': predictor_hidden_dim2,
        'PREDICTOR_HIDDEN_DIM3': predictor_hidden_dim3,
        'OUTPUT_DIM': output_dim
    }

    with open(output_json_path, 'w') as f:
        json.dump(evaluation_metrics, f, indent=4)
    print(f"Evaluation results dumped to {output_json_path}")

    # Create dataframes for plotting directly from the masked (non-NaN original) values
    mon_plot_df = pd.DataFrame({
        'Actual MON': actuals_mon,
        'Predicted MON': predictions_mon
    })

    ron_plot_df = pd.DataFrame({
        'Actual RON': actuals_ron,
        'Predicted RON': predictions_ron
    })

    cn_plot_df = pd.DataFrame({
    'Actual CN': actuals_cn,
    'Predicted CN': predictions_cn
})

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1) # Changed to 1 row, 3 columns
    sns.scatterplot(x='Actual MON', y='Predicted MON', data=mon_plot_df)
    plt.plot([mon_plot_df['Actual MON'].min(), mon_plot_df['Actual MON'].max()],
            [mon_plot_df['Actual MON'].min(), mon_plot_df['Actual MON'].max()],
            'r--', label='Ideal Fit')
    plt.title('MON: Predicted vs. Actual Values')
    plt.xlabel('Actual MON')
    plt.ylabel('Predicted MON')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2) # Changed to 1 row, 3 columns
    sns.scatterplot(x='Actual RON', y='Predicted RON', data=ron_plot_df)
    plt.plot([ron_plot_df['Actual RON'].min(), ron_plot_df['Actual RON'].max()],
            [ron_plot_df['Actual RON'].min(), ron_plot_df['Actual RON'].max()],
            'r--', label='Ideal Fit')
    plt.title('RON: Predicted vs. Actual Values')
    plt.xlabel('Actual RON')
    plt.ylabel('Predicted RON')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3) # Added subplot for CN
    sns.scatterplot(x='Actual CN', y='Predicted CN', data=cn_plot_df)
    plt.plot([cn_plot_df['Actual CN'].min(), cn_plot_df['Actual CN'].max()],
             [cn_plot_df['Actual CN'].min(), cn_plot_df['Actual CN'].max()],
             'r--', label='Ideal Fit')
    plt.title('CN: Predicted vs. Actual Values')
    plt.xlabel('Actual CN')
    plt.ylabel('Predicted CN')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(plots_path)
    plt.close()
    print(f"Predicted vs Actual plots saved to {plots_path}")