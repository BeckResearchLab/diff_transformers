import diff_classifier.features as ft
import diff_classifier.msd as msd
import pandas as pd
import numpy as np
import definitions

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



