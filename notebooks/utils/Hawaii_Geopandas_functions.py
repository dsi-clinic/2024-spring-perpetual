def buffer_in_meters(lon, lat, meters):
    """
    Buffer a geographic point defined in latitude and longitude by a specified distance in meters,
    transforming it to UTM coordinates for buffering, then converting back to WGS84.

    Parameters:
    - lon (float): Longitude of the point in decimal degrees.
    - lat (float): Latitude of the point in decimal degrees.
    - meters (float): The buffer distance in meters.

    Returns:
    - shapely.geometry.polygon.Polygon or None: A polygon representing the buffered area around the point in WGS84 coordinates.
      Returns None if the buffered polygon is empty.

    The function performs coordinate transformations to UTM for accurate distance measurement, buffers the point, and 
    then transforms the buffered polygon's coordinates back to WGS84 for compatibility with geographic applications. This method 
    ensures more accurate distance calculations than buffering directly in geographic coordinates.
    """
    # Transform point to UTM
    x, y = transformer_to_utm.transform(lon, lat)
    
    # Create point and buffer in UTM
    point_utm = Point(x, y)
    buffered_point_utm = point_utm.buffer(meters)
    
    # Convert buffered polygon to WGS84
    if buffered_point_utm.is_empty:
        return None
    else:
        exterior_coords = [(x, y) for x, y in zip(*buffered_point_utm.exterior.coords.xy)]
        transformed_coords = [transformer_to_wgs84.transform(x, y) for x, y in exterior_coords]
        polygon_wgs84 = Polygon(transformed_coords)
        return polygon_wgs84

def calculate_units(row):
    """
    Calculate the estimated number of units in a building based on its square footage and property type.

    Parameters:
    - row (pd.Series): A pandas Series representing a row of a DataFrame, expected to contain
                       the building's property type and its total square footage.

    Returns:
    - float or None: The estimated number of units in the building if the property type is recognized
                     and square footage is available. Returns None if the property type is not recognized
                     or square footage is missing.
    
    The function uses predefined average unit areas for different property types to estimate the number of units.
    If the property type is not in the predefined list or the square footage is not provided, the function returns None.
    """
    
    # Define average areas based on the building type
    average_unit_areas = {
        'Single Family': 2299,
        'Multifamily': 1046,
        'Condo': 592,
        'Apartment': 592,
        'Townhouse': 592,
        'Manufactured': 2000,
        'Land': 2000
    }
    
    # Get the building type from the row
    building_type = row.get('propertyType')
    
    # Look up the average area for the building type
    average_area = average_unit_areas.get(building_type, None)
    
    # Proceed only if the average area is found and 'SQFEET' is valid
    if average_area is not None and 'SQFEET' in row and row['SQFEET'] is not None:
        return row['SQFEET'] / average_area
    else:
        return None
    
    
