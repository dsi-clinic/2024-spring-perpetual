import geopandas as gpd
import geopandas as gpd
from sklearn.neighbors import BallTree
import numpy as np

def check_geodataframe(gdf, name):
    """
    Enhanced check for whether a provided DataFrame is a GeoDataFrame. Verifies the presence of a geometry column,
    and validates the geometries, reporting any invalid ones.

    Args:
        gdf (DataFrame): The DataFrame to check.
        name (str): The descriptive name of the DataFrame for identification in output messages.

    Returns:
        bool: True if the GeoDataFrame is valid, False otherwise.
    """
    if not isinstance(gdf, gpd.GeoDataFrame):
        print(f"Error: '{name}' is not a GeoDataFrame.")
        return False

    if 'geometry' not in gdf.columns:
        print(f"Error: '{name}' does not contain a 'geometry' column.")
        return False

    if not gdf['geometry'].is_valid.all():
        invalid_count = (~gdf['geometry'].is_valid).sum()
        print(f"Warning: {invalid_count} invalid geometries found in '{name}'.")
        # Optionally repair geometries
        gdf['geometry'] = gdf['geometry'].buffer(0)
        print(f"Invalid geometries attempted to be repaired in '{name}'.")
    
    print(f"'{name}' passed all checks as a GeoDataFrame.")
    return True


def get_nearest(src_points, candidates, k_neighbors=1):
    """
    Computes the nearest neighbors for the given source points from the candidate points using a spatial index.

    Args:
        src_points (GeoDataFrame): Source points with geometries.
        candidates (GeoDataFrame): Candidate points with geometries to search within.
        k_neighbors (int): Number of nearest neighbors to find.

    Returns:
        list: A list of tuples containing indices of nearest neighbors and their respective distances.
    """
    # Create a tree from the candidate coordinates
    candidate_coords = np.array(list(zip(candidates.geometry.x, candidates.geometry.y)))
    tree = BallTree(candidate_coords, leaf_size=15, metric='haversine')

    # Query the tree for the nearest neighbors
    src_coords = np.array(list(zip(src_points.geometry.x, src_points.geometry.y)))
    distances, indices = tree.query(src_coords, k=k_neighbors)

    results = [(src_idx, cand_idx, dist) for src_idx, (cand_idx, dist) in enumerate(zip(indices, distances))]
    return results


import matplotlib.pyplot as plt

def plot_geospatial_data(apartments, buildings, connections):
    """
    Plots geospatial data for apartments, buildings, and their connections. Uses color coding and markers
    to differentiate between the data sets and connections.

    Args:
        apartments (GeoDataFrame): GeoDataFrame containing apartment locations and attributes.
        buildings (GeoDataFrame): GeoDataFrame containing building locations and attributes.
        connections (GeoDataFrame): GeoDataFrame containing line geometries connecting nearest apartment-building pairs.

    Returns:
        None: Displays the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    buildings.plot(ax=ax, color='blue', markersize=5, label='Buildings')
    apartments.plot(ax=ax, color='red', markersize=5, label='Apartments')
    connections.plot(ax=ax, color='grey', linestyle='--', linewidth=1, label='Connections')
    
    plt.legend()
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geospatial Relationships')
    plt.show()
