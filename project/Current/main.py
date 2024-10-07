## Instaliation packages that are not normal:
import sys
import definitions as df
import plots
import features as ft
import aws
import norm_def as nd
import data_manipulation as dm
import msd
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch_geometric.nn import GCNConv, GATConv
from torch.nn import BatchNorm1d as BatchNorm
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd


track_from_sql =  sql.data_from_sql("database.db", "SELECT * FROM TRACKMATEDATA")
track_with_frame = df.separate_trajectories(track_from_sql)
trajectories1 = df.separate_data(track_with_frame, False)
trajectories = []
for temp in trajectories1:
    f2 = []
    c = 1
    for t1 in temp:
        f2.append(t1)
        c += 1
        if c == 10:
            break
    trajectories.append(f2)

print(len(trajectories))