import HelperFunctions.definitions as df
import HelperFunctions.plots
import HelperFunctions.features as ft
import HelperFunctions.aws
import HelperFunctions.norm_def as nd
import HelperFunctions.data_manipulation as dm
import HelperFunctions.msd
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
import sys
import os
import HelperFunctions.sql_def as sql
print("Imports worked!")


## <---- DATABASE MPT COLLECTION ----> ##
path_to_database = './data/'
os.chdir(path_to_database)

query = "SELECT * FROM TRACKMATEDATA"
file = "database.db"
track_from_sql =  sql.data_from_sql(file, query)
track_with_frame = df.separate_trajectories(track_from_sql)
trajectories = df.separate_data(track_with_frame, False)
print(f"The length of trajectories from the database at path {path_to_database} is {len(trajectories)}")
