import sqlite3
import random

def data_from_sql(database, command):
    """
    This is a sql connection function, that connects
    to the given database, and runs the command

    Parameters:
    database (str): Database to connect to must be in format of "name.db"
    command (str): sqlite3 command to run within the database

    Returns:
    list: returns the result in a list (each row as one element like (x,y))
    """
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute(command)
    data = cursor.fetchall()
    conn.close()
    return data

def Extractions(trajectories, random_interval=False, max_length=9):
    """
    Extracts subsets from a list of trajectories.
    
    Parameters:
    - trajectories: list of lists containing the trajectory data.
    - random_interval: Boolean, if True, randomly selects the starting point for the subset.
    - max_length: Maximum length of each subset.

    Returns:
    - A list of extracted subsets.
    """
    extracted_trajectories = []

    for temp in trajectories:
        if random_interval:
            if len(temp) > max_length:
                start_index = random.randint(0, len(temp) - max_length)
            else:
                start_index = 0
        else:
            start_index = 0

        end_index = start_index + max_length
        extracted_trajectories.append(temp[start_index:end_index])

    return extracted_trajectories
    
