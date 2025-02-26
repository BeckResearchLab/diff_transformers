



def normalize_data(data, min_x=0, min_y=0, range_x=0, range_y=0):
    """
    Normalizes sequences of points. Automatically determines min and range values if not provided.

    Parameters:
        data (list): List of sequences, where each sequence is a list of (x, y) tuples.
        min_x (float): Minimum x value, determined from data if 0.
        min_y (float): Minimum y value, determined from data if 0.
        range_x (float): Range of x values, determined from data if 0.
        range_y (float): Range of y values, determined from data if 0.

    Returns:
        tuple: A tuple containing the normalized data and the parameters used for normalization.
    """
    # Flatten the list of sequences to a single list of points for min and range calculations
    all_points = [point for seq in data for point in seq if point is not None]
    if range_x == 0 and range_y == 0 and min_x == 0 and min_y == 0:
        all_x = [point[0] for point in all_points]
        all_y = [point[1] for point in all_points]

        min_x = min(all_x)
        max_x = max(all_x)
        min_y = min(all_y)
        max_y = max(all_y)
        range_x = max_x - min_x
        range_y = max_y - min_y

    # Normalize data using the normalize_points_list function
    normalized_data = [normalize_points_list(seq, min_x, min_y, range_x, range_y) for seq in data]

    return normalized_data, min_x, min_y, range_x, range_y



def normalize_point(point, min_x, min_y, range_x, range_y):
    """
    Normalizes a single point based on given minimums and ranges.
    
    Parameters:
        point (tuple): The point (x, y) to normalize.
        min_x (float): Minimum x value from the training data.
        min_y (float): Minimum y value from the training data.
        range_x (float): Range of x values (max_x - min_x) from the training data.
        range_y (float): Range of y values (max_y - min_y) from the training data.

    Returns:
        tuple: A tuple representing the normalized point (normalized_x, normalized_y).
    """
    if point is None:
        return None

    normalized_x = (float(point[0] - min_x) / range_x) if range_x != 0 else 0
    normalized_y = (float(point[1] - min_y) / range_y) if range_y != 0 else 0

    return (normalized_x, normalized_y)

def normalize_points_list(points_list, min_x, min_y, range_x, range_y):
    """
    Normalizes a list of points based on given minimums and ranges.

    Parameters:
        points_list (list): List of tuples, where each tuple represents a point (x, y).
        min_x (float): Minimum x value from the training data.
        min_y (float): Minimum y value from the training data.
        range_x (float): Range of x values (max_x - min_x) from the training data.
        range_y (float): Range of y values (max_y - min_y) from the training data.

    Returns:
        list: A list of tuples, where each tuple represents the normalized point (normalized_x, normalized_y).
    """
    normalized_list = [normalize_point(point, min_x, min_y, range_x, range_y) for point in points_list]
    return normalized_list



def unnormalize_point(normalized_point, min_x, min_y, range_x, range_y):
    """
    Reverts the normalization of a single point based on the original min and range values.
    
    Parameters:
        normalized_point (tuple): The normalized point (normalized_x, normalized_y) to unnormalize.
        min_x (float): Original minimum x value used in normalization.
        min_y (float): Original minimum y value used in normalization.
        range_x (float): Original range of x values (max_x - min_x) used in normalization.
        range_y (float): Original range of y values (max_y - min_y) used in normalization.

    Returns:
        tuple: The unnormalized point (x, y).
    """
    if normalized_point is None:
        return None

    x = normalized_point[0] * range_x + min_x
    y = normalized_point[1] * range_y + min_y

    return (x, y)


def unnormalize_points_list(normalized_points_list, min_x, min_y, range_x, range_y):
    """
    Reverts the normalization for a list of points based on the original min and range values.
    
    Parameters:
        normalized_points_list (list): List of normalized points (tuples) to unnormalize.
        min_x (float): Original minimum x value used in normalization.
        min_y (float): Original minimum y value used in normalization.
        range_x (float): Original range of x values (max_x - min_x) used in normalization.
        range_y (float): Original range of y values (max_y - min_y) used in normalization.

    Returns:
        list: A list of unnormalized points (tuples).
    """
    unnormalized_list = [unnormalize_point(point, min_x, min_y, range_x, range_y) for point in normalized_points_list]
    return unnormalized_list


def unnormalize_data(normalized_data, min_x, min_y, range_x, range_y):
    """
    Reverts the normalization for a dataset of sequences of points based on original min and range values.

    Parameters:
        normalized_data (list): List of sequences, where each sequence is a list of normalized (x, y) tuples.
        min_x (float): Original minimum x value used in normalization.
        min_y (float): Original minimum y value used in normalization.
        range_x (float): Original range of x values (max_x - min_x) used in normalization.
        range_y (float): Original range of y values (max_y - min_y) used in normalization.

    Returns:
        list: A list of sequences, where each sequence contains unnormalized points (tuples).
    """
    unnormalized_data = [unnormalize_points_list(seq, min_x, min_y, range_x, range_y) for seq in normalized_data]
    return unnormalized_data

