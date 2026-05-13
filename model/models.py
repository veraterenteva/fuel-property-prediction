
import torch
import torch.nn as nn

class SMILESEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dims=[256, 64, 16], dropout=0.5, fc_output=12):
        super(SMILESEncoder, self).__init__()

        # The nn.Embedding layer is removed, as input is now sparse vectors.
        # Store vocab_size for use as input_size for the first LSTM.
        self.vocab_size = vocab_size

        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.final_output_dim = fc_output

        self.lstms = nn.ModuleList()
        for i, hidden_dim in enumerate(hidden_dims):
            # The input_dim for the first LSTM is vocab_size from the sparse vector, not embedding_dim.
            input_dim = self.vocab_size if i == 0 else hidden_dims[i-1]
            self.lstms.append(
                nn.LSTM(input_size=input_dim,
                        hidden_size=hidden_dim,
                        num_layers=1,
                        batch_first=True)
            )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dims[-1], fc_output)

    def forward(self, x):
        # Input x is expected to be of shape (batch_size, vocab_size, max_seq_len)
        # Permute to (batch_size, max_seq_len, vocab_size)
        x = x.permute(0, 2, 1) # Now x is (batch, seq_len, input_size)

        for lstm in self.lstms:
            x, _ = lstm(x)  # Output shape: (batch, seq_len, hidden_dim)

        # Take the last time step's output
        x = x[:, -1, :]  # Shape: (batch, hidden_dims[-1])

        # Fully connected layer
        x = self.fc(x)   # Shape: (batch, fc_output)
        return x

class DescriptorEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim3, output_dim)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x

class CombinedModel(nn.Module):
    def __init__(self, 
                 smiles_vocab_size, smiles_embedding_dim, smiles_hidden_dims, smiles_dropout, smiles_linear_output_dim,
                 descriptor_input_dim, descriptor_hidden_dim1, descriptor_hidden_dim2, descriptor_output_dim,
                 predictor_hidden_dim1, predictor_hidden_dim2, predictor_hidden_dim3,
                 output_dim=1):
        super().__init__()

        self.smiles_encoder = SMILESEncoder(
            smiles_vocab_size, smiles_embedding_dim, smiles_hidden_dims, smiles_dropout, smiles_linear_output_dim
        )

        self.descriptor_encoder = DescriptorEncoder(
            descriptor_input_dim, descriptor_hidden_dim1, descriptor_hidden_dim2, descriptor_output_dim
        )

        predictor_input_dim = self.smiles_encoder.final_output_dim + descriptor_output_dim

        self.predictor = Predictor(
            predictor_input_dim, predictor_hidden_dim1, predictor_hidden_dim2, predictor_hidden_dim3, output_dim
        )

    def forward(self, all_smiles_inputs, all_descriptor_inputs, all_molar_fractions, blend_indices):
        # all_smiles_inputs: (total_num_molecules, vocab_size, max_seq_len)
        # all_descriptor_inputs: (total_num_molecules, descriptor_input_dim)
        # all_molar_fractions: (total_num_molecules, 1)
        # blend_indices: (total_num_molecules,) - a tensor indicating which blend each molecule belongs to

        # Encode individual molecules
        smiles_encoded_molecules = self.smiles_encoder(all_smiles_inputs) # (total_num_molecules, smiles_linear_output_dim)
        descriptor_encoded_molecules = self.descriptor_encoder(all_descriptor_inputs) # (total_num_molecules, descriptor_output_dim)

        # Concatenate encoded features for each molecule
        combined_encoded_molecules = torch.cat(
            (smiles_encoded_molecules, descriptor_encoded_molecules), dim=1
        ) # (total_num_molecules, smiles_linear_output_dim + descriptor_output_dim)

        # Apply molar fractions: (total_num_molecules, smiles_linear_output_dim + descriptor_output_dim)
        weighted_features = combined_encoded_molecules * all_molar_fractions

        # Aggregate features for each blend using blend_indices
        # This will sum the weighted features for molecules belonging to the same blend
        # The output will be (batch_size, smiles_linear_output_dim + descriptor_output_dim)
        num_blends = len(torch.unique(blend_indices)) # Get the number of unique blend indices in the current batch
        blend_representations = torch.zeros(
            num_blends,
            combined_encoded_molecules.size(1),
            device=weighted_features.device
        ).scatter_add_(0, blend_indices.unsqueeze(1).expand(-1, weighted_features.size(1)), weighted_features)

        # Predict for each blend
        predictions = self.predictor(blend_representations) # (batch_size, output_dim)

        return predictions
