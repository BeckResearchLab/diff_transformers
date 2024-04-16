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
    # os.chdir(path_to_database)
    # trajectories =  sql.data_from_sql("database.db", "SELECT * FROM TRACKMATEDATA LIMIT 7000")
    # os.chdir(path_to_code)

    # path_to_database = r"C:\Users\alito\Desktop\DeepLearning-Model\diff_transformers\project\data"
    # path_to_code = r"C:\Users\alito\Desktop\DeepLearning-Model\diff_transformers\project"
    # os.chdir(path_to_database)
    # track_from_sql =  sql.data_from_sql("database.db", "SELECT * FROM TRACKMATEDATA LIMIT 7000")
    # os.chdir(path_to_code)
    # track_with_frame = df.separate_trajectories(track_from_sql)
    # frame_data, x_data, y_data = df.separate_data(track_with_frame, True)
    # tracks = df.separate_data(track_with_frame, False)
    # print(tracks)
    # random_trajectories = dm.create_synthetic(100000, 1000, 1000, 0, 0, 5)
    # test2 = dm.create_synthetic(1, 1000, 1000, 0, 0, 5)
    # # random_trajectories = [[(0,0),(1,1),(2,2), (3,3)], [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5)]]
    # #temp = dm.create_synthetic(2, 5, 5, 0, 0, 5)
    # #print(random_trajectories)
    # temp, test  = df.normalize_data(random_trajectories, test2)
