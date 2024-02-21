import definitions as df
import data_manipulation as dm
import sql_def as sql
import os


def main():
    print("main")

if __name__ == "__main__":
    path_to_database = r"C:\Users\alito\Desktop\DeepLearning-Model\diff_transformers\project\data"
    path_to_code = r"C:\Users\alito\Desktop\DeepLearning-Model\diff_transformers\project"
    os.chdir(path_to_database)
    trajectories =  sql.data_from_sql("database.db", "SELECT * FROM TRACKMATEDATA WHERE slide=1 AND video=1")
    # print(trajectories)
    os.chdir(path_to_code)
    data_with_frame = df.separate_trajectories(trajectories)
    # result = []
    print(df.find_max_length(data_with_frame))

    cutoff = df.find_min_length(data_with_frame)
    end = df.find_max_length(data_with_frame)

    final_result = []
    while (cutoff <= end):
        temp_res = []
        for track in data_with_frame:
            if len(track) < cutoff:
                continue
            new_track = df.listTrim(track, cutoff)
            temp = dm.list_calculate(new_track)
            temp_res.append(temp)

        final_result.append(df.calculate_avg(temp_res))
    print(final_result)
    #data_manipulation.list_calculate([1,2,3 ], [1,2 , 3], [1,2, 3])