
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch_geometric.nn as gnn
from torch_geometric.nn.conv import GCNConv

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

class EquivariantGraphNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(EquivariantGraphNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x

class EquivariantTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(EquivariantTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer = nn.Transformer(
            d_model=model_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers
        )
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, src, src_mask=None):
        src = self.embedding(src)
        src = src.permute(1, 0, 2) 
        
        transformer_output = self.transformer(src, src, src_mask, src_mask)
        
        output = self.fc_out(transformer_output)
        return output.permute(1, 0, 2)




class SelfSupervisedModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, model_dim, num_heads, num_layers, output_dim):
        super(SelfSupervisedModel, self).__init__()
        self.gnn = EquivariantGraphNN(in_channels, hidden_channels, out_channels)
        self.transformer = EquivariantTransformer(out_channels, model_dim, num_heads, num_layers, output_dim)

    def forward(self, x, edge_index, src_mask=None, tgt_mask=None):
        gnn_output = self.gnn(x, edge_index) 
        gnn_output = gnn_output.permute(1, 0, 2)
        
        transformer_output = self.transformer(gnn_output, src_mask)
        transformer_output = transformer_output.permute(1, 0, 2) 
        
        return transformer_output[:, -1, :]


