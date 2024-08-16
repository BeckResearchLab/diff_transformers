import diff_classifier.features as ft
import diff_classifier.msd as msd
import pandas as pd
import numpy as np
import definitions
import random 

def list_calculate (trajectory):
    frame, data_x, data_y = definitions.separate_data(trajectory)
    frames = len(frame)
    data1 = {'Frame': np.linspace(1, frames, frames),
             'X': data_x,
             'Y': data_y}
    dframe = pd.DataFrame(data=data1)
    dframe = msd.msd_calc(dframe, frames)
    res, trash = ft.alpha_calc(dframe)
    return res

def create_synthetic(size, maxX=1000, maxY=1000, minX=0, minY=0, len=30):
    """
    Creates n trajectories that are fake

    """
    data = []
    for _ in range(size):
        length = random.randint(5, len)
        length = len
        x = random.randint(minX, maxX)
        y = random.randint(minY, maxY)
        m = random.randint(1, 10)
        r = random.choice([True, False])
        rotation_val = random.randint(0, 360)

        trajectory = definitions.random_trajectory_straight(length, x, y, m, r)
        data.append(trajectory)

    return data



