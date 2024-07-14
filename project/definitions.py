## package imports ##
import numpy as np
import math
import random 
import matplotlib.pyplot as plt
import torch

## local package import ##
import data_manipulation as dm
import plots

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

    

def normalize_data(data, min_x=0, min_y=0, range_x=0, range_y=0):
    # (x, y) -> needs to be the data, otherwise i gotta rewrite possibly
    # can assume with none
    all_points = [point for seq in data for point in seq if point is not None]
    if range_x or range_y == 0 or min_x == 0 or min_y == 0:
        all_x = [point[0] for point in all_points]
        all_y = [point[1] for point in all_points] 

        min_x = min(all_x)
        max_x = max(all_x)
        min_y = min(all_y)
        max_y = max(all_y)
        range_x = max_x - min_x
        range_y = max_y - min_y


    # Normalize data
    normalized_data = []
    for seq in data:
        normalized_seq = [(float(point[0] - min_x) / range_x, float(point[1] - min_y) / range_y) if point is not None else None for point in seq]
        normalized_data.append(normalized_seq)

    return normalized_data, min_x, min_y, range_x, range_y

def save_data(filename, data):
    with open(filename, 'w') as file:
        for item in data:
            file.write(f"{item}\n")


def split_test_train(data, percentOfSplit=90):
    """
    percent of split, like 90-10 or 60-40 and what not given from 0 to 100 as val
    """
    test = []
    train = []

    percent = percentOfSplit / 100
    train_len = math.floor(len(data) * percent)
    for tracks in data:
        if train_len > 0:
            train.append(tracks)
        else:
            test.append(tracks)
        train_len = train_len - 1
    return train, test


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


def prepare_data_for_transformer(normalized_data, masked_points):
    src_data = []
    tgt_data = []
    src_masks = []

    #max_len = max(len(trajectory) for trajectory in normalized_data)

    for i, trajectory in enumerate(normalized_data):
        src_seq = []
        src_mask = []
        tgt_seq = []

        masked_value = masked_points[i]

        for j, point in enumerate(trajectory):
            if point is not None:
                src_seq.append(point)
                tgt_seq.append(point)
                src_mask.append((False, False))
            else:
                src_seq.append((0,0)) ## puting it as None, does not let us convert to tensor, what should we do intead?
                src_mask.append((True, True))
                tgt_seq.append(masked_value)

        # while len(src_seq) < max_len:
        #     src_seq.append((0.0, 0.0))
        #     src_mask.append(False)

        src_data.append(src_seq)
        tgt_data.append(tgt_seq)
        src_masks.append(src_mask)
    print("hello")
    src_data_tensor = torch.tensor(src_data, dtype=torch.float32)
    tgt_data_tensor = torch.tensor(tgt_data, dtype=torch.float32)
    src_masks_tensor = torch.tensor(src_masks, dtype=torch.bool)

    return src_data_tensor, tgt_data_tensor, src_masks_tensor
