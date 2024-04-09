import definitions as df
import data_manipulation as dm
import sql_def as sql
import os
import numpy as np
import File_Def as FD

import matplotlib.pyplot as plt
import plots
def main():
    print("main")

def calculate_mass_alpha(data_with_frame):
    cutoff = df.find_min_length(data_with_frame)
    end = df.find_max_length(data_with_frame)
    
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

            new_track = df.listTrim(track, cutoff)
            temp = dm.list_calculate(new_track)
            if (isinstance(temp, np.float64)):
                temp_res.append(temp)    

        final_result.append(df.calculate_avg(temp_res))
        cutoff = cutoff + 1

        plot(final_result, "final_result.png")
        FD.save_data("finalResult.txt", final_result)
        FD.save_data("Full_n.txt", df.calculate_avg(full_list))

if __name__ == "__main__":
    path_to_database = r"C:\Users\alito\Desktop\DeepLearning-Model\diff_transformers\project\data"
    path_to_code = r"C:\Users\alito\Desktop\DeepLearning-Model\diff_transformers\project"
    # os.chdir(path_to_database)
    # trajectories =  sql.data_from_sql("database.db", "SELECT * FROM TRACKMATEDATA LIMIT 7000")
    # os.chdir(path_to_code)

    random_trajectories = dm.create_synthetic(2, 5, 5, 0, 0, 5)
    # random_trajectories = [[(0,0),(1,1),(2,2), (3,3)], [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5)]]
    #temp = dm.create_synthetic(2, 5, 5, 0, 0, 5)
    print(random_trajectories)
    temp = df.normalize_data(random_trajectories)
    print(temp)
    normalized_data = df.mask_point_at_index(temp, 2)
    print(normalized_data)
    plots.plot_syn_norm(normalized_data)
    # seq_len = len(normalized_data[0])
    # d_model = 128 

    # pos_encoding = df.positional_encoding(seq_len, d_model)



