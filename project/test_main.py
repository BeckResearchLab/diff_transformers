import File_Def
import definitions as df
# data = [1, 2, 3, 4, 5]
# File_Def.save_data("hi.txt", data)
jack = df.random_trajectory_straight(5, 2, 2, 3, False)
rose = df.random_trajectory_straight(5, 2, 2, 3, True, 180)
print(jack)
print(rose)