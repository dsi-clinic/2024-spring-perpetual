def check_roomCounts(data):
    """
    Prints the formatted address and room count for each property in the data list that contains a room count.
    
    Iterates through a list of property dictionaries, checks for the presence of a room count in the property's features,
    and prints out the property's formatted address alongside the room count if it exists. Additionally, it counts
    the number of properties that contain a room count.
    
    Parameters:
    - data (list of dict): A list of dictionaries, where each dictionary represents a property and contains
      a 'formattedAddress' key and a 'features' dictionary that may include a 'roomCount' key.
    
    Returns:
    - int: The number of properties that contain a room count.
    """
    count_contained = 0
    for item in data:
        formattedAddress = item.get('formattedAddress')
        features = item.get('features',{})
        roomCount = features.get('roomCount', None)
        if roomCount is not None:
            print(f"Address: {formattedAddress}, Room Count: {roomCount}")
            count_contained += 1
    return count_contained


def check_unitCounts(data):
    """
    Prints the formatted address and unit count for each property in the data list that contains a unit count.
    
    Iterates through a list of property dictionaries, checks for the presence of a unit count in the property's features,
    and prints out the property's formatted address alongside the unit count if it exists. Additionally, it counts
    the number of properties that contain a unit count.
    
    Parameters:
    - data (list of dict): A list of dictionaries, where each dictionary represents a property and contains
      a 'formattedAddress' key and a 'features' dictionary that may include a 'unitCount' key.
    
    Returns:
    - int: The number of properties that contain a unit count.
    """
    count_contained = 0
    for item in data:
        formattedAddress = item.get('formattedAddress')
        features = item.get('features',{})
        unitCount = features.get('unitCount', None)
        if unitCount is not None:
            print(f"Address: {formattedAddress}, Unit Count: {unitCount}")
            count_contained += 1
    return count_contained


def check_features(data):
    """
    Prints the features of each property in the data list that contains additional feature information.
    
    Iterates through a list of property dictionaries and prints the 'features' dictionary for each property,
    if it exists. The 'features' dictionary is assumed to contain additional details about the property.
    
    Parameters:
    - data (list of dict): A list of dictionaries, where each dictionary represents a property and
      may contain a 'features' dictionary with additional property details.
    
    Returns:
    - None
    """
    for item in data:
        features = item.get('features', {})  # Returns {} if 'features' is not found
        if features:
            print(features)