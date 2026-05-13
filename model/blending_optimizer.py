
import scipy.optimize
import numpy as np
import torch
import random

def find_best_blend(target_ron, target_mon, available_smiles, smiles_map, descriptors_map, model, device, l1_molar_ratio_penalty=0.0):
    """
    Finds the best fuel blend composition to match target RON and MON using optimization.

    Args:
        target_ron (float): Desired RON value.
        target_mon (float): Desired MON value.
        available_smiles (list): List of SMILES strings for pure components available for blending.
        smiles_map (dict): Pre-computed sparse vectors for SMILES strings.
        descriptors_map (dict): Pre-computed scaled descriptors for SMILES strings.
        model (nn.Module): The trained CombinedModel.
        device (torch.device): The device (cpu/cuda) the model is on.
        l1_molar_ratio_penalty (float): Strength of L1 regularization on molar fractions to encourage sparsity.

    Returns:
        tuple: (optimal_composition_dict, predicted_ron, predicted_mon, final_loss)
               Returns (None, None, None, None) if optimization fails.
    """

    num_components = len(available_smiles)

    # Prepare inputs for the model for all available components
    all_smiles_inputs_components = torch.stack([smiles_map[s] for s in available_smiles]).to(device)
    all_descriptor_inputs_components = torch.stack([descriptors_map[s] for s in available_smiles]).to(device)

    model.eval() # Ensure model is in evaluation mode

    def predict_properties_for_blend(molar_fractions_np):
        """
        Internal function to predict RON/MON for a given blend composition.
        Takes numpy array of molar fractions, returns numpy array of predictions.
        """
        molar_fractions_tensor = torch.FloatTensor(molar_fractions_np).unsqueeze(1).to(device) # Shape (num_components, 1)

        # For a single blend, all components belong to blend 0
        blend_indices_tensor = torch.zeros(num_components, dtype=torch.long).to(device)

        with torch.no_grad():
            predictions = model(
                all_smiles_inputs_components,
                all_descriptor_inputs_components,
                molar_fractions_tensor,
                blend_indices_tensor
            )
            # Predictions are [MON, RON] based on BlendDataset target order
            return predictions.cpu().squeeze().numpy()

    def objective_function(molar_fractions_np):
        """
        Objective function to minimize.
        Calculates the squared error between predicted and target MON/RON,
        plus an L1 regularization term on molar fractions for sparsity.
        """
        predicted_properties = predict_properties_for_blend(molar_fractions_np)
        predicted_mon, predicted_ron = predicted_properties[0], predicted_properties[1]

        # Mean Squared Error
        loss = (target_mon - predicted_mon)**2 + (target_ron - predicted_ron)**2

        # L1 regularization on molar fractions to encourage sparsity
        l1_penalty = l1_molar_ratio_penalty * np.sum(np.abs(molar_fractions_np))

        return loss + l1_penalty

    # Robust Initial Guess Logic
    initial_guess = np.ones(num_components) * (1.0 / num_components)
    initial_guess = initial_guess + np.random.normal(0, 0.001, num_components)
    min_molar_ratio = 1e-6
    initial_guess = np.maximum(initial_guess, min_molar_ratio)
    initial_guess = initial_guess / np.sum(initial_guess)

    bounds = [(min_molar_ratio, 1.0) for _ in range(num_components)]
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}) # Sum of molar fractions must be 1

    result = scipy.optimize.minimize(
        objective_function,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'disp': False, 'ftol': 1e-4} # Increased tolerance for more robust convergence
    )

    if result.success:
        # Normalize result.x to ensure it sums to 1.0, accounting for potential minor floating point inaccuracies
        optimal_molar_fractions = result.x / np.sum(result.x)
        predicted_properties_optimal = predict_properties_for_blend(optimal_molar_fractions)
        predicted_mon_optimal, predicted_ron_optimal = predicted_properties_optimal[0], predicted_properties_optimal[1]

        optimal_composition_dict = {
            available_smiles[i]: frac for i, frac in enumerate(optimal_molar_fractions)
        }
        return optimal_composition_dict, predicted_ron_optimal, predicted_mon_optimal, result.fun
    else:
        return None, None, None, None

def find_k_component_blend(target_ron, target_mon, k_components, all_available_smiles, smiles_map, descriptors_map, model, device, num_trials=100, molar_fraction_threshold=1e-4):
    """
    Finds the best fuel blend composition with exactly 'k_components' components
    by performing a stochastic search and optimization.

    Args:
        target_ron (float): Desired RON value.
        target_mon (float): Desired MON value.
        k_components (int): The desired number of components in the final blend.
        all_available_smiles (list): List of all possible SMILES strings for pure components.
        smiles_map (dict): Pre-computed sparse vectors for SMILES strings.
        descriptors_map (dict): Pre-computed scaled descriptors for SMILES strings.
        model (nn.Module): The trained CombinedModel.
        device (torch.device): The device (cpu/cuda) the model is on.
        num_trials (int): Number of random initial selections of k components to try.
        molar_fraction_threshold (float): Minimum molar fraction to consider a component "present".

    Returns:
        tuple: (optimal_composition_dict, predicted_ron, predicted_mon, final_loss)
               Returns (None, None, None, None) if no successful blend is found.
    """
    best_overall_loss = float('inf')
    best_overall_composition = None
    best_overall_pred_ron = None
    best_overall_pred_mon = None

    if k_components <= 0:
        print("Number of components (k) must be positive.")
        return None, None, None, None
    if k_components > len(all_available_smiles):
        print(f"Cannot select {k_components} components from a pool of {len(all_available_smiles)} available SMILES.")
        return None, None, None, None

    model.eval()

    print(f"Starting stochastic search for a {k_components}-component blend over {num_trials} trials...")

    for trial in range(num_trials):
        if num_trials >= 10 and trial % (num_trials // 10) == 0:
             print(f"Trial {trial+1}/{num_trials}...")
        elif num_trials < 10:
            print(f"Trial {trial+1}/{num_trials}...")

        # Randomly select k_components unique SMILES
        current_smiles_subset = random.sample(all_available_smiles, k_components)

        # Optimize molar fractions for this subset using find_best_blend with L1_molar_ratio_penalty=0.0
        # because we are *fixing* the number of components, not encouraging sparsity.
        optimal_composition_dict, pred_ron, pred_mon, current_loss = find_best_blend(
            target_ron,
            target_mon,
            current_smiles_subset,
            smiles_map,
            descriptors_map,
            model,
            device,
            l1_molar_ratio_penalty=0.0 # No L1 penalty when k components are already chosen
        )

        if optimal_composition_dict is not None and current_loss is not None and current_loss < best_overall_loss:
            best_overall_loss = current_loss
            best_overall_pred_ron = pred_ron
            best_overall_pred_mon = pred_mon
            best_overall_composition = {
                s: f for s, f in optimal_composition_dict.items() if f > molar_fraction_threshold
            }

    if best_overall_composition:
        print(f"Stochastic search complete. Best blend found with a loss of {best_overall_loss:.4f}.")
        return best_overall_composition, best_overall_pred_ron, best_overall_pred_mon, best_overall_loss
    else:
        print("Stochastic search completed, but no successful blend was found.")
        return None, None, None, None
