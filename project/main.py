import definitions as df
import data_manipulation as dm
import sql_def as sql
import os
import numpy as np

import matplotlib.pyplot as plt
import plots

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import math
import torch.nn.functional as F
import model as md


def main():
    print("main")


def compute_accuracy(predictions, targets, threshold=0.01):
    distances = torch.sqrt(torch.sum((predictions - targets) ** 2, dim=1))
    accurate_predictions = distances < threshold
    accuracy = torch.mean(accurate_predictions.float()) * 100
    return accuracy.item()

def prepare_data_for_transformer(normalized_data, masked_points):
    src_data = []
    tgt_data = []
    src_masks = []

    for i, trajectory in enumerate(normalized_data):
        src_seq = []
        tgt_seq = []
        src_mask = []

        for _, point in enumerate(trajectory):
            if point is not None:
                src_seq.append(point)
                tgt_seq.append(point)
                src_mask.append(False)
            else:
                src_seq.append((0.0, 0.0))   ## 0 for now, we could change it ig
                tgt_seq.append(masked_points[i])
                src_mask.append(True)

        src_data.append(src_seq)
        tgt_data.append(tgt_seq)
        src_masks.append(src_mask)

    src_data_tensor = torch.tensor(src_data, dtype=torch.float32)
    tgt_data_tensor = torch.tensor(tgt_data, dtype=torch.float32)
    src_masks_tensor = torch.tensor(src_masks, dtype=torch.bool)
    #src_masks_tensor = src_masks_tensor.transpose(0, 1)

    return src_data_tensor, tgt_data_tensor, src_masks_tensor


class BasicTrajectoryModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BasicTrajectoryModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    path_to_database = r"C:\Users\alito\Desktop\DeepLearning-Model\diff_transformers\project\data"
    path_to_code = r"C:\Users\alito\Desktop\DeepLearning-Model\diff_transformers\project"


    total_size = 12000
    max_x = 1000
    max_y = 1000
    min_x = 0
    min_y = 0
    track_len = 20

    trajectories = dm.create_synthetic(total_size, max_x, max_y, min_x, min_y, track_len)

    train_norm, test_norm = df.split_test_train(trajectories)
    train, min_x, min_y, range_x, range_y = df.normalize_data(train_norm)
    test, min_x, min_y, range_x, range_y = df.normalize_data(test_norm, min_x, min_y, range_x, range_y)
    train_data, train_masked_point = df.mask_point_at_index(train, 6)
    test_data, test_masked_point = df.mask_point_at_index(test, 6)

    print(len(train_data))
    print(len(test_masked_point))


    src_data_tensor, tgt_data_tensor, src_masks_tensor = df.prepare_data_for_transformer(train_data, train_masked_point)

## DEBUG
    print(src_data_tensor)
    print("src_data_tensor shape:", src_data_tensor.shape)
    print("tgt_data_tensor shape:", tgt_data_tensor.shape)
    print("src_masks_tensor shape:", src_masks_tensor.shape)


    input_dim = 2
    hidden_dim = 64                                                                                   
    output_dim = 2
    rate = 0.0001

    #model = md.SelfSupervisedModel(in_channels=3, out_channels=128, input_dim=128, model_dim=256, num_heads=8, num_layers=6, output_dim=3)
    model = md.LSTMModel(2, 32, 2, 1)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=rate) ## w
    dataset = TensorDataset(src_data_tensor, tgt_data_tensor, src_masks_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


    ## TEMP
    n_iters = 40
    learning_rate = 0.001
    batch_size = 32
    threshold = 0.01

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_values = []
    accuracy_values = []

    for epoch in range(40):
        total_loss = 0
        model.train()  # Set the model to training mode
        for src, tgt, mask in dataloader:
            optimizer.zero_grad()
            output = model(src)
            masked_output = output  # output[:, 6, :]
            loss = loss_function(masked_output, tgt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        loss_values.append(avg_loss)
        print(f'Epoch {epoch+1}/{n_iters}, Loss: {avg_loss}')

        # Evaluation phase
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            t_src_data_tensor, t_tgt_data_tensor, t_src_masks_tensor = df.prepare_data_for_transformer(test_data, test_masked_point)
            print(t_tgt_data_tensor[2])
            predictions = model(t_src_data_tensor)
            print(predictions[2])
            accuracy = compute_accuracy(predictions, t_tgt_data_tensor, threshold=0.01)
            print(accuracy)
            accuracy_values.append(accuracy)
            print(f'Accuracy after Epoch {epoch+1}/{n_iters}: {accuracy:.2f}%')

    # Plotting loss and accuracy
    epochs = range(1, 40 + 1)

    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_values, 'b', label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy_values, 'r', label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()