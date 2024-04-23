
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch_geometric.nn as gnn

## From 455
class BasicTrajectoryModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicTrajectoryModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        return x
    
## From 455
class EnhancedTrajectoryModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EnhancedTrajectoryModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.activation(x)
        return x
    
## From google
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x) 
        out = hn.squeeze(0)
        out = self.fc(out)
        return out


class TrajectoryTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_encoder_layers, dropout=0.1):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Embedding(10, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(d_model, input_dim)

    def forward(self, src, src_key_padding_mask):
        src = self.input_embedding(src) + self.position_embedding(torch.arange(0, src.size(1), device=src.device))
        if src_key_padding_mask is not None and src_key_padding_mask.size(1) != src.size(0):
            src_key_padding_mask = src_key_padding_mask.transpose(0, 1)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        return self.output_layer(output[:, 0, :])



