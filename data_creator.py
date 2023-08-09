import numpy as np

def create_data_model(filename, number_of_vehicles):
    """Stores the data for the problem."""

    data = {}
    # The matrix is divided by 10 for ease of calculation
    data['distance_matrix'] = (np.load(filename)/10).astype(int)
    data['num_vehicles'] = int(number_of_vehicles)
    data['depot'] = 0
    return data