import sqlite3

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