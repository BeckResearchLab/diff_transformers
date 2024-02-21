## Sample for testing purposes ##
import numpy as np
import math
import random
import features as ft

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

            modified_sublist[index] = (0,0) ## how should we do this masking?  change to none later

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

def random_trajectory_straight(length, x=0, y=0, m=1, r=False, rotation_val=0):
    """
    returns a straight line trajectory in a x-y matrix

    Parameters:
    length (list): number of points in a list
    x (int): starting x position of line
    y (int): starting y position of line
    m (int): amount of space between the points
    r (boolean): if rotation is enabled or not
    theta (int): degree of rotation (for testing purposes)
    Returns:
    list: a list of points that make up a line
    """
    if (length == 0):
        return []
    

    points = [(x, y)]
    theta = 0

    if(rotation_val == 0): 
        if (r):
            theta = math.radians(random.uniform(0, 360))
    else:
        print("we are here?")
        theta = math.radians(rotation_val)
        print(theta)

    for i in range(1, length):
        new_x = (m * i) + x
        new_y = y
        if(r or rotation_val != 0):
            rotated_x = new_x * math.cos(theta) - new_y * math.sin(theta)
            rotated_y = new_x * math.sin(theta) + new_y * math.cos(theta)
            points.append((rotated_x, rotated_y))
        else:
            points.append((new_x, new_y))
    
    return points

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



def separate_data (data):
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
## cut the length array to a fixed length

## diff clasifer - diff clasifer - features.py -> alpha calc.


## reocnsutrction loss -?
## compression 0 loss, aaproximatiation 
# alpha being the trajectory of the data

## alpha mean squared derivation coeeftion , alpha about 1 is increasing.
## 