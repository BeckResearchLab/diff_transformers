import sql_def as sd
import os 

path_to_database = r"C:\Users\alito\Desktop\DeepLearning-Model\diff_transformers\project\data"
path_to_code = r"C:\Users\alito\Desktop\DeepLearning-Model\diff_transformers\project"
os.chdir(path_to_database)
data =  sd.data_from_sql("database.db", "SELECT * FROM TRACKMATEDATA WHERE ")

for value in data:
    print(value)