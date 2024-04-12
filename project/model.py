import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class BasicTrajectoryModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BasicTrajectoryModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class EnhancedTrajectoryModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EnhancedTrajectoryModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.25)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
def compute_accuracy(predictions, targets, threshold=0.1):
    distances = torch.sqrt(torch.sum((predictions - targets) ** 2, dim=1))
    accurate_predictions = distances < threshold
    accuracy = torch.mean(accurate_predictions.float()) * 100
    return accuracy.item()

def evaluate_model(model, test_src_data_tensor, test_tgt_data_tensor, test_src_masks_tensor):
    model.eval()
    total_loss = 0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        outputs = model(test_src_data_tensor)
        outputs = outputs.view(test_tgt_data_tensor.shape)

        masked_outputs = outputs[test_src_masks_tensor]
        masked_targets = test_tgt_data_tensor[test_src_masks_tensor]

        total_loss = criterion(masked_outputs, masked_targets)
        accuracy = compute_accuracy(masked_outputs, masked_targets)

    return total_loss.item(), accuracy