## package imports ##
import numpy as np
import math
import random 
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data, Batch

## local package import ##
import HelperFunctions.data_manipulation as dm
import HelperFunctions.plots
import HelperFunctions.msd as msd
import HelperFunctions.features as ft
import pandas as pd


def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def separate_trajectories(data):
    """
    This method sepeartes the givne trajecories from all points
    to specific trajectories based on track ID

    Parameters:
    data (list): the list of all points collected

    Returns:
    list: a new double list that will have each trajectory sperated. 

    Requires:
    data to be in format of: [(track id, frame, x, y), ...]
    """
    trajectories = {}
    # {Inv: len[data_0] = len(trajectories[0 to i]) + len(data[i to n])}
    for pos in data:
        exp, sl, vid, track_id, frame, x, y = pos
        key = f"{exp}_{sl}_{vid}_{track_id}"
        if key not in trajectories:
            trajectories[key] = []

        trajectories[key].append((frame, x, y))
    return list(trajectories.values())


def mask_point_at_index(data, index):
    """
    This method grabs a list of trajectories, and
    makes a specific position of  point that is given 
    to be set to be "masked"

    Parameters:
    data (list): the list of all trajectories
    index: The index to mask


    Returns:
    list: a new double list that will have a specific point in each trajectory mask
    list: returns the original value 

    Requires:
    data to be in format of: [(track id, frame, x, y), ...]
    """
    masked_data = []
    masked_points = []

    for trajectory in data:

        modified_sublist = trajectory.copy()

        if 0 <= index < len(trajectory):

            masked_points.append(trajectory[index])

            modified_sublist[index] = None ## how should we do this masking?  change to none later

            masked_data.append(modified_sublist)

    return masked_data, masked_points

def find_min_length(data):
    """
    finds the minimum length of a given list of list

    Parameters:
    data (list): the list of all trajectories

    Returns:
    int: minimum length of the list
    """
    if (len(data) == 0):
        return 0

    min_length = len(data[0])

    for lst in data:
        if len(lst) < min_length:
            min_length = len(lst)

    return min_length

def find_max_length(data):
    """
    finds the maximum length of a given list of list

    Parameters:
    data (list): the list of all trajectories

    Returns:
    int: maximum length of the list
    """
    if (len(data) == 0):
        return 0

    max_length = len(data[0])

    for lst in data:
        if len(lst) > max_length:
            max_length = len(lst)

    return max_length



def rotate_point(x, y, angle_degrees, origin=(0, 0)):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    Parameters:
    x, y (float): the point to rotate.
    angle_degrees (float): the angle of rotation in degrees.
    origin (tuple): the point around which to rotate.

    Returns:
    tuple: the rotated point.
    """
    angle_radians = math.radians(angle_degrees)
    ox, oy = origin
    qx = ox + math.cos(angle_radians) * (x - ox) - math.sin(angle_radians) * (y - oy)
    qy = oy + math.sin(angle_radians) * (x - ox) + math.cos(angle_radians) * (y - oy)
    return qx, qy

def random_trajectory_straight(length, x=0, y=0, m=1, r=False, rotation_val=0, decimals=6):
    """
    Returns a straight line trajectory in an x-y matrix.

    Parameters:
    length (int): number of points in the trajectory.
    x (int): starting x position of line.
    y (int): starting y position of line.
    m (int): amount of space between the points.
    r (boolean): if rotation is enabled or not.
    rotation_val (int): degree of rotation (for testing purposes).

    Returns:
    list: a list of points that make up a line.
    """
    trajectory = []
    rotation_val = random.randint(0, 360)
    for i in range(length):
        point = (x + i * m, y)
        if r:
            point = rotate_point(point[0], point[1], rotation_val, origin=(x, y))
        # Round the point to the specified number of decimals
        point = (round(point[0], decimals), round(point[1], decimals))
        trajectory.append(point)
    return trajectory

def listTrim (data, index):
    """
    
    """
    output_list = data[:index]
    return output_list

def cut_frame_data (data):

    for index in range(len(data)):
        trajectory = data[index]
        for element in range(len(trajectory)):
            t1, t2, t3 = trajectory[element]
            trajectory[element] = (t2, t3)

    return data

def calculate_avg (data):
    sum = 0
    for values in data:
        sum += values

    return sum / len(data)

def separate_data_helper (data):
    """
    This method sepeartes the givne trajecory into 3 arrrays of
    frame, x and y

    Parameters:
    data (list): the list of all points collected

    Returns:
    list: a new double list that will have each trajectory sperated. 

    Requires:
    data to be in format of: [(track id, frame, x, y), ...]
    """
    frame = []
    x = []
    y = []
    for index in range(len(data)):
        t1, t2, t3 = data[index]
        frame.append(t1)
        x.append(t2)
        y.append(t3)

    return frame, x, y

def separate_data (data, type=False):
    frame = []
    x = []
    y = []
    for track in data:
        frame_temp, x_temp, y_temp = separate_data_helper(track)
        frame.append(frame)
        x.append(x_temp)
        y.append(y_temp)
    if(type):
        return frame, x, y
    else:
        track = []
        for index in range(len(x)):
            x_tracks = x[index]
            y_tracks = y[index]
            temp = []
            for index_track in range(len(x_tracks)):
                temp.append((x_tracks[index_track], y_tracks[index_track]))
            track.append(temp)
        return track
    
def plot_points (data, name):
    index = np.linspace(1, len(data), num=len(data))
    plt.plot(data, 'o')
    plt.savefig(name)


def save_data(filename, data):
    with open(filename, 'w') as file:
        for item in data:
            file.write(f"{item}\n")


def split_data(data, train_percent=70, val_percent=20):
    """
    Splits data into training, validation, and testing sets based on provided percentages.
    Parameters:
    - data: The dataset to split.
    - train_percent: Percentage of data to use for training.
    - val_percent: Percentage of data to use for validation.
    Returns:
    - train: Training set.
    - val: Validation set.
    - test: Testing set.
    """
    train = []
    val = []
    test = []

    train_len = math.floor(len(data) * train_percent / 100)
    val_len = math.floor(len(data) * val_percent / 100)
    test_len = len(data) - train_len - val_len

    for i, tracks in enumerate(data):
        if i < train_len:
            train.append(tracks)
        elif i < train_len + val_len:
            val.append(tracks)
        else:
            test.append(tracks)

    return train, val, test



def calculate_mass_alpha(data_with_frame):
    cutoff = find_min_length(data_with_frame)
    end = find_max_length(data_with_frame)
    
    full_list = []
    for track in data_with_frame:    
        temp = dm.list_calculate(track)
        if (isinstance(temp, np.float64)):
            full_list.append(temp)                      
                              
    final_result = []
    while (cutoff <= end):
        temp_res = []
        print("started on " + str(cutoff))

        for track in data_with_frame:

            if len(track) < cutoff:
                continue

            new_track = listTrim(track, cutoff)
            temp = dm.list_calculate(new_track)
            if (isinstance(temp, np.float64)):
                temp_res.append(temp)    

        final_result.append(calculate_avg(temp_res))
        cutoff = cutoff + 1

        plots.plot(final_result, "final_result.png")
        save_data("finalResult.txt", final_result)
        save_data("Full_n.txt", calculate_avg(full_list))

def prepare_data_for_transformer_with_padding(normalized_data, masked_points, MAX_VALUE_TRAJECTORIES_LENGTH):
    src_data = []
    tgt_data = []
    src_masks = []

    for i, trajectory in enumerate(normalized_data):
        src_seq = []
        src_mask = []
        src_tgt = []

        masked_value = masked_points[i]

        for j, point in enumerate(trajectory):
            if point is not None:
                src_seq.append(point)
                src_mask.append(False)  # Not masked
                src_tgt.append(point)
            else:
                src_seq.append((0, 0))  # Placeholder for masked point
                src_mask.append(True)   # Masked
                src_tgt.append(masked_value)

        src_data.append(src_seq)
        tgt_data.append(src_tgt)
        src_masks.append(src_mask)
    padding_mask_tensor = torch.tensor(pad_mask(src_data, MAX_VALUE_TRAJECTORIES_LENGTH), dtype=torch.bool)
    src_data = pad_track(src_data, MAX_VALUE_TRAJECTORIES_LENGTH)
    tgt_data = pad_track(tgt_data, MAX_VALUE_TRAJECTORIES_LENGTH)
    src_masks = pad_track(src_masks, MAX_VALUE_TRAJECTORIES_LENGTH, pad_value=False)

    src_data_tensor = torch.tensor(src_data, dtype=torch.float32)
    tgt_data_tensor = torch.tensor(tgt_data, dtype=torch.float32)
    src_masks_tensor = torch.tensor(src_masks, dtype=torch.bool)

    return src_data_tensor, tgt_data_tensor, src_masks_tensor, padding_mask_tensor



def compute_accuracy(predictions, targets, threshold=0.01):
    distances = torch.sqrt(torch.sum((predictions - targets) ** 2, dim=1))
    accurate_predictions = distances < threshold
    accuracy = torch.mean(accurate_predictions.float()) * 100
    return accuracy.item()

def shift_to_origin(data):
    """
    Shifts sequences of points so that each sequence starts at (0, 0).

    Parameters:
        data (list): List of sequences, where each sequence is a list of (x, y) tuples.

    Returns:
        list: A list of sequences where each sequence is shifted to start at (0, 0).
    """
    shifted_data = []
    for seq in data:
        if seq[0] is not None:
            first_point = seq[0]
            shifted_seq = [(point[0] - first_point[0], point[1] - first_point[1]) for point in seq if point is not None]
            shifted_data.append(shifted_seq)
    return shifted_data

def pad_track(data, max, pad_value=(0,0)):
    """
    pads to the given max length

    Parameters:
    data (list): the list of all trajectories
    max (int): the maximum length of the list

    Returns:
    int: minimum length of the list
    """
    if (len(data) == 0):
        return 0
    final_data = []
    for lst in data:
        length = len(lst);
        data_lst = [];
        for i in range(length):
            data_lst.append(lst[i])
        for i in range(length, max):
            data_lst.append(pad_value)
        final_data.append(data_lst)

    return final_data

def pad_mask(data, max):
    """
    pads blueprint to the given max length

    Parameters:
    data (list): the list of all trajectories
    max (int): the maximum length of the list

    Returns:
    int: minimum length of the list
    """
    mask = []
    for lst in data:
        padding_mask = [False] * len(lst) + [True] * (max - len(lst))
        mask.append(padding_mask)
    print(len(mask))
    print(len(mask[0]))
    return mask


def unpad(data, mask):
    """
    Takes in list
    """
    temp_pred = []
    for elem, mask_pad in zip(data, mask):
        temp_track = []
        for elem1, mask_elem1 in zip(elem, mask_pad):
            if not mask_elem1:
              temp_track.append(elem1)
        temp_pred.append(temp_track)
    return temp_pred


def calculate_accuracy_with_tolerance(data, tg, bp, tolerance=10):
    correct = 0
    total = 0

    for i, (seq, mask_seq, target_seq) in enumerate(zip(data, bp, tg)):
        target_index = 0
        for j, (mask, pred) in enumerate(zip(mask_seq, seq)):
            if mask:
                if target_index < len(target_seq):
                    target = target_seq
                    total += 1
                    if abs(pred[0] - target[0]) <= tolerance and abs(pred[1] - target[1]) <= tolerance:
                        correct += 1
                    target_index += 1
                else:
                    print("Warning")
    if total == 0:
        return 0  # Avoid division by zero
    accuracy = (correct / total) * 100
    return accuracy

def create_edge_index_basic(num_points, connectivity_range=2):
    """
    The basic version of the edge index creation. This only connects a single point to its connectivity_rangeth
    nodes. It stops connecting before 0 and after max points.
    Parameters:
        num_points (number): number of points in a single trajectory.
        connectivity_range (number): number of edges from a node to each side.

    Returns:
        edge_index: a list that has the connections
        edge_weight: a list with the weights of each edge
    """
    row = []
    col = []
    weights = []

    for i in range(num_points):
        for j in range(max(0, i-connectivity_range), min(num_points, i+connectivity_range+1)):
            if i != j:
                row.append(i)
                col.append(j)
                weight = 1 / abs(i - j)
                weights.append(weight)

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    return edge_index, edge_weight


def create_edge_index_with_distance(points):
    points = torch.tensor(points, dtype=torch.float32)  # Convert list to tensor

    num_points = points.shape[0]
    row, col = torch.meshgrid(torch.arange(num_points), torch.arange(num_points), indexing="ij")

    mask = row != col  # Exclude self-loops
    row, col = row[mask], col[mask]  # Apply mask to remove diagonal elements

    # Compute Euclidean distances efficiently
    diff = points[row] - points[col]
    edge_weight = torch.norm(diff, dim=1)

    edge_index = torch.stack([row, col], dim=0)  # Shape (2, num_edges)

    return edge_index, edge_weight

def prepare_graph_data_for_batch(normalized_data, masked_points):
    data_list = []
    src_mask_list = []

    for i, trajectory in enumerate(normalized_data):
        src_seq = []
        src_mask = []

        for point in trajectory:
            # Ensure point is converted to a tuple or array for comparison
            point = tuple(point)
            # Check if the point is (0, 0)
            if not np.all(np.isclose(point, (0, 0))):  # Valid points
                src_seq.append(point)
                src_mask.append(False)  # Not masked
            else:
                src_seq.append((0, 0))  # Placeholder for masked points
                src_mask.append(True)   # Masked point

        # Convert the trajectory to a graph
        edge_index, edge_weight = create_edge_index_with_distance(src_seq)

        # Convert data to tensors
        src_seq_tensor = torch.tensor(src_seq, dtype=torch.float32)
        src_mask_tensor = torch.tensor(src_mask, dtype=torch.bool)

        # Create a PyTorch Geometric Data object for the graph
        graph_data = Data(x=src_seq_tensor, edge_index=edge_index, edge_weight=edge_weight)
        data_list.append(graph_data)
        src_mask_list.append(src_mask_tensor)

    # Stack the masks into a batch of shape [batch_size, sequence_length]
    src_mask_batch = torch.stack(src_mask_list)

    return Batch.from_data_list(data_list), src_mask_batch

def dataFixer(AllTracks, torch=True):
    """
    data = [[],[],[]]
    """
    All_tracks_records = []
    for oneTrack in AllTracks:
        x_value = []
        y_value = []
        frame = np.linspace(1, len(oneTrack), len(oneTrack));
        for onePoint in oneTrack:
            onePointList = onePoint
            if torch:
                onePointList = onePoint.tolist();
            x_value.append(onePointList[0])
            y_value.append(onePointList[1])
        data_one_track = {'Frame': frame, 'X': x_value, 'Y': y_value}
        All_tracks_records.append(data_one_track)
    return All_tracks_records


def calculate_mse(output_list, tgt_list):
    if len(output_list) != len(tgt_list):
        raise ValueError("output_list and tgt_list must have the same length")
    
    total_error = 0
    n = len(output_list)
    
    for pred, target in zip(output_list, tgt_list):
        total_error += (pred - target) ** 2
    
    mse = total_error / n if n > 0 else 0
    return mse



def diff_accuracy(Predictions, padding_mask, target):

    pred_track = unpad(Predictions, padding_mask);
    tgt_track = unpad(target, padding_mask);

    dataForMSD_PRED = dataFixer(pred_track, False)
    dataForMSD_TGT = dataFixer(tgt_track, False)

    output_list = []
    tgt_list = []
    for data in dataForMSD_PRED:
        dframe = pd.DataFrame(data=data)
        dframe = msd.msd_calc(dframe, len(data))
        res, trash = ft.alpha_calc(dframe)
        output_list.append(res)
    for data1 in dataForMSD_TGT:
        dframe1 = pd.DataFrame(data=data1)
        dframe1 = msd.msd_calc(dframe1, len(data))
        res1, trash1 = ft.alpha_calc(dframe1)
        tgt_list.append(res1)
    
    return calculate_weighted_mse(output_list, tgt_list)


    