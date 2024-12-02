from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
import torch.nn as nn
import torch
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

class EGNNLayer(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(EGNNLayer, self).__init__(aggr='mean')
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + 1, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.edge_mlp(edge_input)

    def update(self, aggr_out, x):
        node_input = torch.cat([x, aggr_out], dim=-1)
        return self.node_mlp(node_input)

class EGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout_rate=0.3):
        super(EGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(EGNNLayer(in_channels, hidden_channels, hidden_channels))
            elif i == num_layers - 1:
                self.layers.append(EGNNLayer(hidden_channels, hidden_channels, out_channels))
            else:
                self.layers.append(EGNNLayer(hidden_channels, hidden_channels, hidden_channels))

        self.norm = nn.LayerNorm(out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_weight.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.norm(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# class EquivariantGraphNN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.3):
#         super(EquivariantGraphNN, self).__init__()
#         self.conv1 = GATConv(in_channels, hidden_channels, heads=8, concat=True)
#         self.norm1 = nn.LayerNorm(hidden_channels * 8)
#         self.dropout1 = nn.Dropout(dropout_rate)
#         self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True)
#         self.norm2 = nn.LayerNorm(hidden_channels * 8)
#         self.dropout2 = nn.Dropout(dropout_rate)
#         self.conv3 = GATConv(hidden_channels * 8, out_channels, heads=8, concat=False)
#         self.norm3 = nn.LayerNorm(out_channels)
#         self.dropout3 = nn.Dropout(dropout_rate)

#     def forward(self, data):
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
#         x = F.relu(self.norm1(self.conv1(x, edge_index)))
#         x = self.dropout1(x)
#         x = F.relu(self.norm2(self.conv2(x, edge_index)))
#         x = self.dropout2(x)
#         x = self.norm3(self.conv3(x, edge_index))
#         x = self.dropout3(x)
#         return x
# Working Transformer
# class EquivariantTransformer(nn.Module):
#     def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout_rate=0.1):
#         super(EquivariantTransformer, self).__init__()
#         self.embedding = nn.Linear(input_dim, model_dim)
#         self.positional_encoding = PositionalEncoding(model_dim)
#         self.transformer = nn.Transformer(d_model=model_dim, nhead=num_heads, num_encoder_layers=num_layers,
#                                           num_decoder_layers=num_layers, dropout=dropout_rate)
#         self.fc_out = nn.Linear(model_dim, output_dim)
#         self.mask_token = nn.Parameter(torch.zeros(1, 1, input_dim))

#     def forward(self, src, src_mask=None):
#         src = self.embedding(src)
#         #src = self.positional_encoding(src)
#         src = src.permute(1, 0, 2)
#         if src_mask is not None:
#             mask_token = self.mask_token.expand(src.size(0), src.size(1), -1)
#             src = torch.where(src_mask.T.unsqueeze(-1), mask_token, src)
#         transformer_output = self.transformer(src, src, src_key_padding_mask=src_mask)
#         output = self.fc_out(transformer_output)
#         return output.permute(1, 0, 2)


class EquivariantTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout_rate=0.1):
        super(EquivariantTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        self.transformer = nn.Transformer(d_model=model_dim, nhead=num_heads, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dropout=dropout_rate)
        self.fc_out = nn.Linear(model_dim, output_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, input_dim))

    def forward(self, src, src_mask=None, padding_mask=None):
        src = self.embedding(src)
        src = self.positional_encoding(src)
        src = src.permute(1, 0, 2)  # (batch, seq_len, dim) -> (seq_len, batch, dim)
        if src_mask is not None:
            mask_token = self.mask_token.expand(src.size(0), src.size(1), -1)
            src = torch.where(src_mask.T.unsqueeze(-1), mask_token, src)

        # Apply the padding mask here
        transformer_output = self.transformer(src, src, src_key_padding_mask=padding_mask)

        output = self.fc_out(transformer_output)
        return output.permute(1, 0, 2)

class SelfSupervisedModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, model_dim, num_heads, num_layers, output_dim):
        super(SelfSupervisedModel, self).__init__()
        # Use EGNN for graph-based representation
        self.egnn = EGNN(in_channels, hidden_channels, out_channels)
        # Use transformer for sequence modeling
        self.transformer = EquivariantTransformer(out_channels, model_dim, num_heads, num_layers, output_dim)

    def forward(self, data_batch, src_mask=None, padding_mask=None):
        # GNN output: pass the graph data through EGNN
        gnn_output = self.egnn(data_batch)
        batch_size = data_batch.num_graphs

        # Calculate sequence length
        sequence_length = gnn_output.size(0) // batch_size

        # Reshape GNN output to (batch_size, sequence_length, out_channels)
        gnn_output = gnn_output.view(batch_size, sequence_length, -1)

        # Pass the GNN output and masks to the transformer
        transformer_output = self.transformer(gnn_output, src_mask=src_mask, padding_mask=padding_mask)

        # # Truncate the transformer output based on the padding mask
        # if padding_mask is not None:
        #     # Transpose padding mask to match the transformer output shape (batch_size, seq_len)
        #     padding_mask = padding_mask.cpu().numpy()
        #     # Keep only the non-padded parts of the output
        #     non_padded_output = [output[~mask] for output, mask in zip(transformer_output, padding_mask)]
        #     # Return a padded tensor of truncated output
        #     return torch.nn.utils.rnn.pad_sequence(non_padded_output, batch_first=True)
        # else:
        return transformer_output



