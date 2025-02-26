
import torch.nn as nn

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

