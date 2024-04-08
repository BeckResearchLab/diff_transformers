## Sample for testing purposes ##
import numpy as np
import math
import diff_classifier.features as ft
import matplotlib.pyplot as plt
import random 

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
    for i in range(length):
        point = (x + i * m, y)
        if r:
            rotation_val = random.randint(0, 360)
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


def plot_points (data, name):
    print(len(data))
    index = np.linspace(1, len(data), num=len(data))
    print(len(index))
    plt.plot(index, data, 'o')
    plt.savefig(name)

    