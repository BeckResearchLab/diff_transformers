import HelperFunctions.definitions as df
import HelperFunctions.plots
import HelperFunctions.features as ft
import HelperFunctions.aws
import HelperFunctions.norm_def as nd
import HelperFunctions.data_manipulation as dm
import HelperFunctions.msd
import HelperFunctions.sql_def as sql
import Models.Advanced_Models as AM
import Models.Basic_Models as BM
import Loss.loss as LM

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GCNConv, GATConv
from torch.nn import BatchNorm1d as BatchNorm
import torch.optim.lr_scheduler as lr_scheduler

import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import sys
import os

## <-- GLOBAL VARIABLES --> ##
MAX_VALUE_TRAJECTORIES_LENGTH = 25
RANDOM_TRACK = False

RANDOM_MASKING = False
MASKING_INDEX = 5

HYPERTUNNING = False
FREQ_PRINT = 1

INCLUDE_TESTIING = True
INCLUDE_GRAPHS = True

## <---- CUDA CHECK ----> ##
if torch.cuda.is_available():
    print("CUDA is available!")
    device = torch.device("cuda")
else:
    print("CUDA is not available.")
    device = torch.device("cpu")
    
PATH_TO_BASE = "/content/diff_transformers/project"


os.chdir(PATH_TO_BASE);
## <---- DATABASE MPT COLLECTION ----> ##
path_to_database = PATH_TO_BASE + '/data/'
os.chdir(path_to_database)
query = "SELECT * FROM TRACKMATEDATA"
file = "database.db"
track_from_sql =  sql.data_from_sql(file, query)
track_with_frame = df.separate_trajectories(track_from_sql)
trajectories_raw = df.separate_data(track_with_frame, False)
trajectories = sql.Extractions(trajectories_raw, False, MAX_VALUE_TRAJECTORIES_LENGTH)
trajectories_random = sql.Extractions(trajectories_raw, True, MAX_VALUE_TRAJECTORIES_LENGTH)
max_len = df.find_max_length(trajectories)
max_len_random = df.find_max_length(trajectories_random)
print(f"--> The length of trajectories from the database at path {path_to_database} is {len(trajectories_raw)}")
print(f"--> The length of trajectories Extracted is {len(trajectories)} with max length: {max_len}")
print(f"--> The length of trajectories Randomly Extracted is {len(trajectories_random)} with max length: {max_len_random} ")

os.chdir(PATH_TO_BASE);

## <---- TRAJECTORIES TO TENSORS ----> ##
shifted_data = df.shift_to_origin(trajectories)
train_norm, val_norm, test_norm = df.split_data(shifted_data, train_percent=70, val_percent=10)
train, min_x, min_y, range_x, range_y = nd.normalize_data(train_norm)
val, _, _, _, _ = nd.normalize_data(val_norm, min_x, min_y, range_x, range_y)
test, _, _, _, _ = nd.normalize_data(test_norm, min_x, min_y, range_x, range_y)
if RANDOM_MASKING == False:
  train_data, train_masked_point = df.mask_point_at_index(train, MASKING_INDEX)
  val_data, val_masked_point = df.mask_point_at_index(val, MASKING_INDEX)
  test_data, test_masked_point = df.mask_point_at_index(test, MASKING_INDEX)
else:
  ## WILL ADD LATER AFTER INTIAL TESTING ON HYAK
  print("WILL ADD LATER AFTER INTIAL TESTING ON HYAK, this will crash")

src_data_tensor, tgt_data_tensor, src_masks_tensor, src_padding_mask_tensor = df.prepare_data_for_transformer_with_padding(train_data, train_masked_point, MAX_VALUE_TRAJECTORIES_LENGTH)
val_src_data_tensor, val_tgt_data_tensor, val_src_masks_tensor, val_src_padding_mask_tensor = df.prepare_data_for_transformer_with_padding(val_data, val_masked_point, MAX_VALUE_TRAJECTORIES_LENGTH)
test_src_data_tensor, test_tgt_data_tensor, test_src_masks_tensor, test_src_padding_mask_tensor = df.prepare_data_for_transformer_with_padding(test_data, test_masked_point, MAX_VALUE_TRAJECTORIES_LENGTH)

print(f"The shape of src input is {src_masks_tensor.shape}")
print(f"The shape of src input is {val_src_data_tensor.shape}")
print(f"The shape of src input is {test_src_data_tensor.shape}")


## <---- MODEL SETUP NORMAL----> ##
if HYPERTUNNING == False:
    #PARAMS
    in_channels = 2
    hidden_channels = 256
    out_channels = 256
    model_dim = 256
    num_heads = 4
    num_layers = 4
    output_dim = 2
    learning_rate = 0.0001

    #MODEL
    model = AM.SelfSupervisedModel(in_channels, hidden_channels, out_channels, model_dim, num_heads, num_layers, output_dim)
    model = model.to(device)

    # LOSS, OPTIMIZER, SCHEDUILUER
    loss_function = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    #DATALOADERS
    train_dataset = TensorDataset(src_data_tensor, tgt_data_tensor, src_masks_tensor, src_padding_mask_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = TensorDataset(val_src_data_tensor, val_tgt_data_tensor, val_src_masks_tensor, val_src_padding_mask_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    test_dataset = TensorDataset(test_src_data_tensor, test_tgt_data_tensor, test_src_masks_tensor, test_src_padding_mask_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Starting Training NORMAL with Train batch size of {train_dataloader.batch_size} and Val batch size of {val_dataloader.batch_size}")

    ## <---- MODEL ITTERATION NORMAL----> ##
    n_iters = 3
    loss_values = []
    val_loss_values = []

    print(f"Starting Normal Training with following information: Number of Iter: {n_iters}")
    for epoch in range(n_iters):
        model.train()
        total_loss = 0
        torch.cuda.empty_cache()
        
        # TRAINING
        for src, tgt, mask, padding_mask in train_dataloader:
            src, tgt, mask, padding_mask = src.to(device), tgt.to(device), mask.to(device), padding_mask.to(device)
            graph_batch, src_mask_batch = df.prepare_graph_data_for_batch(src.cpu().numpy(), mask.cpu().numpy())
            graph_batch = graph_batch.to(device)
            src_mask_batch = src_mask_batch.to(device)
            optimizer.zero_grad()
            output = model(graph_batch, src_mask=src_mask_batch, padding_mask=padding_mask)
            loss = loss_function(output, tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            del src, tgt, mask, padding_mask, graph_batch, src_mask_batch, output, loss
            torch.cuda.empty_cache()
        avg_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_loss)
        
        # VALIDATION 
        model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for src, tgt, mask, padding_mask in val_dataloader:
                src, tgt, mask, padding_mask = src.to(device), tgt.to(device), mask.to(device), padding_mask.to(device)
                graph_batch, src_mask_batch = df.prepare_graph_data_for_batch(src.cpu().numpy(), mask.cpu().numpy())
                graph_batch = graph_batch.to(device)
                src_mask_batch = src_mask_batch.to(device)
                optimizer.zero_grad()
                output = model(graph_batch, src_mask=src_mask_batch, padding_mask=padding_mask)
                val_loss = loss_function(output, tgt)

                val_total_loss += val_loss.item()
        avg_val_loss = val_total_loss / len(val_dataloader)
        val_loss_values.append(avg_val_loss)
        
        if epoch % FREQ_PRINT == 0:
            print(f'Epoch {epoch+1}/{n_iters}, Training Loss: {avg_loss:.8f}, Validation Loss: {avg_val_loss:.8f}')
        scheduler.step(avg_val_loss)

    # TESTING
    if INCLUDE_TESTIING:
        model.eval()
        predictionList = []
        with torch.no_grad():
            c = 0
            acc = 0
            for src, tgt, mask, padding_mask in test_dataloader:
                c += 1
                src, tgt, mask, padding_mask = src.to(device), tgt.to(device), mask.to(device), padding_mask.to(device)
                graph_batch, src_mask_batch = df.prepare_graph_data_for_batch(src.cpu().numpy(), mask.cpu().numpy())
                graph_batch = graph_batch.to(device)
                src_mask_batch = src_mask_batch.to(device)

                predictions = model(graph_batch, src_mask=src_mask_batch, padding_mask=padding_mask)
                tgt_converted = [[(float(coord[0]), float(coord[1])) for coord in sequence] for sequence in tgt]

                err = df.diff_accuracy(predictions.cpu().tolist(), padding_mask.cpu().tolist(), tgt_converted)
                print(err)
                acc += err

                predictionList.append(predictions)
        print(f'Accuracy after Epoch {n_iters}: {acc/c:.2f}%')

    if INCLUDE_GRAPHS:
        print("Starting Graphs")
        test_x = [point[0] for point in test[0]]
        test_y = [point[1] for point in test[0]]
        predictions_x = predictionList[0][0][:, 0].cpu().numpy()
        predictions_y = predictionList[0][0][:, 1].cpu().numpy()
        plt.figure(figsize=(10, 5))
        plt.scatter(test_x, test_y, color='blue', label='Test Data')
        plt.scatter(predictions_x, predictions_y, color='red', label='Predictions', marker='x')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Test Data vs. Predictions')
        plt.legend()
        plt.savefig("predictedVSActual.png")
        
        epochs = range(1, n_iters + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss_values, 'b', label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_loss_values, 'r', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig("Loss.png")

## <---- MODEL HYPERTUNNING ----> ##       
else:
    print("Starting with HyperParameterTunning. This will take a long time depending on chosen parameters")