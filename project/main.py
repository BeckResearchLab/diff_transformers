import definitions as df
import data_manipulation as dm
import sql_def as sql
import os
import numpy as np
import File_Def as FD

import matplotlib.pyplot as plt
import random

def main():
    print("main")


def plot_syn():
    random_trajectories = dm.create_synthetic(200)
    for trajectory in random_trajectories:
        x_values = [point[0] for point in trajectory]  # Extracting x-values
        y_values = [point[1] for point in trajectory]  # Extracting y-values
        plt.plot(x_values, y_values)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectories')
    plt.grid(True)
    plt.show()

def plot(data):
    df.plot_points(data, "fig1.png")

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
    # data_with_frame = df.separate_trajectories(trajectories)
    # print(data_with_frame)
    # for trajectory in data_with_frame:
    #     x_values = [point[1] for point in trajectory]  # Extracting x-values
    #     y_values = [point[2] for point in trajectory]  # Extracting y-values
    #     plt.plot(x_values, y_values)

    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Trajectories')
    # plt.grid(True)
    # plt.show()
    plot_syn()
