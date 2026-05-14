import torch
import pandas as pd
from rdkit import Chem

from model.models import CombinedModel
from model.blending_optimizer import find_k_component_blend
from model.data_preprocessing import (
    get_smiles_vocabulary,
    get_top_n_descriptors,
    preprocess_data,
    smiles_to_sparse_vectors,
    calculate_rdkit_descriptors,
)
import model.config as config


class FuelModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading data...")

        self.df_pure = pd.read_csv("data/pure_for_mix.csv")

        self.df_mix = pd.read_csv("data/mix_combined_cn.csv")

        not_smiles = ("index", "Mixture name", "Dataset", "RON", "MON")

        self.all_smiles = list(
            filter(lambda x: x not in not_smiles, self.df_mix.columns)
        )

        print(f"Loaded {len(self.all_smiles)} SMILES")
        (
            self.char_to_idx,
            self.idx_to_char,
            self.vocab_size,
            self.max_seq_len,
        ) = get_smiles_vocabulary(self.all_smiles)

        (self.selected_descriptor_names, _) = get_top_n_descriptors(
            self.df_pure, top_n_descriptors=config.TOP_N_DESCRIPTORS, seed=config.SEED
        )

        (self.smiles_map, self.descriptors_map, _, _, self.scaler) = preprocess_data(
            df_mix=self.df_mix,
            df_pure_components=self.df_pure,
            all_smiles_in_mix=self.all_smiles,
            selected_descriptor_names=self.selected_descriptor_names,
            char_to_idx=self.char_to_idx,
            max_seq_len=self.max_seq_len,
        )

        print("Building model...")

        self.model = CombinedModel(
            smiles_vocab_size=self.vocab_size,
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
            output_dim=config.OUTPUT_DIM,
        )

        self.model.load_state_dict(
            torch.load("best_model.pth", map_location=self.device)
        )

        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully")

    def ensure_smiles_exists(self, smiles):
        if smiles in self.smiles_map:
            return None
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return {
                "valid": False,
                "smiles": smiles,
                "error": "Invalid SMILES"
            }
        # safe add
        self.smiles_map[smiles] = smiles_to_sparse_vectors(
            [smiles],
            self.char_to_idx,
            self.max_seq_len
        )[0]

        descriptors = calculate_rdkit_descriptors(smiles)
        descriptor_df = pd.DataFrame([descriptors])
        scaled = self.scaler.transform(
            descriptor_df[self.selected_descriptor_names]
        )
        self.descriptors_map[smiles] = torch.FloatTensor(scaled[0])

        return {"valid": True}

    def validate_fractions(self, blend):
        for c in blend:
            try:
                f = float(str(c["fraction"]).replace(",", "."))
            except:
                return {
                    "error": "Invalid fraction format",
                    "smiles": c["smiles"]
                }
            if f <= 0:
                return {
                    "error": "Fraction must be > 0",
                    "smiles": c["smiles"]
                }
        return None

    # Forward prediction
    def get_properties_from_mixture(self, blend):
        err = self.validate_fractions(blend)

        if err:
            return err

        invalid_smiles = []

        for component in blend:

            res = self.ensure_smiles_exists(
                component["smiles"]
            )

            if isinstance(res, dict) and res.get("valid") is False:
                invalid_smiles.append(res["smiles"])

        if len(invalid_smiles) > 0:
            return {
                "error": "Invalid SMILES detected",
                "invalid_smiles": invalid_smiles
            }

        smiles_list = [x["smiles"] for x in blend]

        fractions = torch.FloatTensor(
            [x["fraction"] for x in blend]
        ).unsqueeze(1)

        smiles_inputs = torch.stack([
            self.smiles_map[s] for s in smiles_list
        ])

        descriptor_inputs = torch.stack([
            self.descriptors_map[s] for s in smiles_list
        ])

        blend_indices = torch.zeros(len(smiles_list), dtype=torch.long)

        with torch.no_grad():

            prediction = self.model(
                smiles_inputs.to(self.device),
                descriptor_inputs.to(self.device),
                fractions.to(self.device),
                blend_indices.to(self.device)
            )

        prediction = prediction.cpu().numpy()[0]

        return {
            "MON": float(prediction[0]),
            "RON": float(prediction[1])
        }

    # Inverse problem
    def get_mixture_from_properties(
            self, target_ron, target_mon, k=4, num_trials=100
    ):

        (composition, pred_ron, pred_mon, loss) = find_k_component_blend(
            target_ron=target_ron,
            target_mon=target_mon,
            k_components=k,
            all_available_smiles=self.all_smiles,
            smiles_map=self.smiles_map,
            descriptors_map=self.descriptors_map,
            model=self.model,
            device=self.device,
            num_trials=num_trials,
        )

        if composition is None:
            return {"error": "Blend not found"}

        blend = []

        for smiles, fraction in composition.items():
            blend.append({"smiles": smiles, "fraction": float(fraction)})

        return {
            "predicted_RON": float(pred_ron),
            "predicted_MON": float(pred_mon),
            "loss": float(loss),
            "blend": blend,
        }
