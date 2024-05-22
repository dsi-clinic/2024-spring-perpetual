import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import matplotlib.pyplot as plt
import warnings
import spaghetti
import folium
import requests
import os
from dotenv import load_dotenv
from pathlib import Path
from sklearn.cluster import KMeans
from spopt.locate import MCLP
from pulp import PULP_CBC_CMD
from scipy.spatial import cKDTree

# Suppress warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

# Define parameters
CLIENT_COUNT = 200
FACILITY_COUNT = 125
SERVICE_RADIUS = 30
P_FACILITIES = 10

# Load the environment variables from the .env file
ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Access the Mapbox API token
MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")


def load_and_clean_data(api_data_path, foot_traffic_path, large_apartments_path, small_apartments_path):
    """
    Load and clean the datasets for the analysis.

    Parameters:
    - api_data_path (str): Path to the API data file.
    - foot_traffic_path (str): Path to the foot traffic data file.
    - large_apartments_path (str): Path to the large apartments data file.
    - small_apartments_path (str): Path to the small apartments data file.

    Returns:
    - gpd.GeoDataFrame: Cleaned GeoDataFrame for all apartments.
    - gpd.GeoDataFrame: Cleaned GeoDataFrame for foot traffic points.
    - gpd.GeoDataFrame: Cleaned GeoDataFrame for large apartments.
    """
    # Load API Data
    api_data = pd.read_json(api_data_path)
    geometry = [Point(xy) for xy in zip(api_data['longitude'], api_data['latitude'])]
    hilo_all_gdf = gpd.GeoDataFrame(api_data, geometry=geometry, crs="EPSG:4326")

    # Load building foot-traffic data
    foot = pd.read_parquet(foot_traffic_path, engine='pyarrow')

    large_apartments_NJ = gpd.read_file(large_apartments_path)
    small_apartment_NJ = gpd.read_file(small_apartments_path)

    # Drop duplicates
    large_apartments_NJ = large_apartments_NJ.drop_duplicates(subset=['latitude', 'longitude'])
    small_apartment_NJ = small_apartment_NJ.drop_duplicates(subset=['latitude', 'longitude'])

    # Concatenate the two GeoDataFrames
    large_apartments_NJ = pd.concat([large_apartments_NJ, small_apartment_NJ])

    return hilo_all_gdf, foot, large_apartments_NJ


def summarize_clusters(gdf):
    """
    Summarizes the clusters in a GeoDataFrame that includes a 'cluster' label column derived from K-means clustering.
    It now also calculates the total 'raw_visit_counts' for each cluster as weights.

    Parameters:
    - gdf (gpd.GeoDataFrame): GeoDataFrame with 'latitude', 'longitude' for foot traffic points,
                              a 'cluster' column, and 'raw_visit_counts' indicating the volume of visits.
    
    Returns:
    - pd.DataFrame: DataFrame where each row represents a cluster with columns for the
                    number of points in the cluster, total visit counts, the cluster label, 
                    and the centroid geometry.
    """
    # Ensure the 'cluster' column exists
    if 'cluster' not in gdf.columns:
        raise ValueError("GeoDataFrame must include a 'cluster' column")

    # Group by cluster label
    grouped = gdf.groupby('cluster')

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'num_points': grouped.size(),  # Count points per cluster
        'total_visit_counts': grouped['raw_visit_counts'].sum(),  # Sum of visit counts per cluster
        'centroid': grouped.apply(lambda x: Point(x['longitude'].mean(), x['latitude'].mean()))  # Calculate centroid of each cluster
    }).reset_index()

    # Rename columns appropriately
    summary_df.columns = ['cluster_label', 'num_points', 'total_visit_counts', 'geometry']

    # Convert centroids to GeoDataFrame
    summary_df = gpd.GeoDataFrame(
        summary_df, 
        geometry='geometry',
        crs='EPSG:4326'  # Set the coordinate reference system to WGS 84
    )

    return summary_df


def cluster_foot_traffic(foot_traffic_gdf, n_clusters=200):
    """
    Perform K-means clustering on foot traffic data.

    Parameters:
    - foot_traffic_gdf (gpd.GeoDataFrame): GeoDataFrame containing foot traffic data.
    - n_clusters (int): Number of clusters for K-means.

    Returns:
    - gpd.GeoDataFrame: GeoDataFrame with cluster labels assigned.
    """
    coords = foot_traffic_gdf[['latitude', 'longitude']]
    coords = coords.fillna(coords.mean())  # Fill NaNs with mean of each column
    coords.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinities with NaN
    coords.fillna(coords.mean(), inplace=True)  # Fill new NaNs potentially created

    # Running K-means
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(coords.values)

    # Assign clusters back to the GeoDataFrame
    foot_traffic_gdf['cluster'] = clusters

    return foot_traffic_gdf


def clean_coordinates(gdf):
    """
    Ensure all geometries are valid and finite.

    Parameters:
    - gdf (gpd.GeoDataFrame): GeoDataFrame to clean.

    Returns:
    - gpd.GeoDataFrame: Cleaned GeoDataFrame.
    """
    # Remove any rows with None or NaN geometries
    gdf = gdf[gdf['geometry'].notna()]
    # Ensure all coordinates are finite
    gdf = gdf[gdf['geometry'].apply(lambda geom: np.isfinite(geom.x) and np.isfinite(geom.y))]
    return gdf


def calculate_cost_matrix(demand_gdf, supply_gdf, radius=None, mapbox_access_token=None):
    """
    Calculate the cost matrix for demand and supply points, either using local computation or Mapbox API.

    Parameters:
    - demand_gdf (gpd.GeoDataFrame): GeoDataFrame of demand points.
    - supply_gdf (gpd.GeoDataFrame): GeoDataFrame of supply points.
    - radius (float): Service radius for local computation (ignored if using Mapbox).
    - mapbox_access_token (str): Mapbox access token for API usage (optional).

    Returns:
    - np.array: Cost matrix.
    """
    demand_coords = np.array(list(demand_gdf.geometry.apply(lambda geom: (geom.x, geom.y))))
    supply_coords = np.array(list(supply_gdf.geometry.apply(lambda geom: (geom.x, geom.y))))

    if mapbox_access_token:
        return generate_mapbox_cost_matrix(demand_coords, supply_coords, mapbox_access_token)
    
    tree = cKDTree(supply_coords)
    demand_indices = np.arange(len(demand_coords))
    supply_indices = [tree.query_ball_point(point, r=radius) for point in demand_coords]

    cost_matrix = np.zeros((len(demand_indices), len(supply_coords)), dtype=bool)
    for demand_idx, nearby_supply_indices in zip(demand_indices, supply_indices):
        cost_matrix[demand_idx, nearby_supply_indices] = True

    return cost_matrix


def generate_mapbox_cost_matrix(demand_coords, supply_coords, mapbox_access_token, max_coords=25):
    """
    Generate the cost matrix using the Mapbox API.

    Parameters:
    - demand_coords (list): List of demand coordinates.
    - supply_coords (list): List of supply coordinates.
    - mapbox_access_token (str): Mapbox access token.
    - max_coords (int): Maximum number of coordinates per API request.

    Returns:
    - np.array: Cost matrix.
    """
    def chunk_coordinates(coords, chunk_size):
        for i in range(0, len(coords), chunk_size):
            yield coords[i:i + chunk_size]

    def get_cost_matrix(client_chunk, facility_chunk, mapbox_access_token):
        all_coords = client_chunk + facility_chunk
        coord_str = ';'.join([f"{lon},{lat}" for lon, lat in all_coords])
        url = f"https://api.mapbox.com/directions-matrix/v1/mapbox/driving/{coord_str}?annotations=duration&access_token={mapbox_access_token}"
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code} from Mapbox API")
            print(response.text)
            return np.full((len(client_chunk), len(facility_chunk)), np.inf)

        matrix_data = response.json()
        if 'durations' not in matrix_data:
            print("Error: 'durations' key not found in response")
            print(matrix_data)
            return np.full((len(client_chunk), len(facility_chunk)), np.inf)
        
        return np.array(matrix_data['durations'])

    client_chunks = list(chunk_coordinates(demand_coords, max_coords // 2))
    facility_chunks = list(chunk_coordinates(supply_coords, max_coords // 2))

    cost_matrix = np.full((len(demand_coords), len(supply_coords)), np.inf)

    for i, client_chunk in enumerate(client_chunks):
        for j, facility_chunk in enumerate(facility_chunks):
            chunk_cost_matrix = get_cost_matrix(client_chunk, facility_chunk, mapbox_access_token)
            num_clients = len(client_chunk)
            num_facilities = len(facility_chunk)
            start_client_idx = i * (max_coords // 2)
            start_facility_idx = j * (max_coords // 2)
            cost_matrix[start_client_idx:start_client_idx + num_clients, start_facility_idx:start_facility_idx + num_facilities] = chunk_cost_matrix[:num_clients, num_clients:num_clients + num_facilities]

    return cost_matrix


def calculate_weights_and_cost_matrix(client_points, facility_points, service_radius, mapbox_access_token=None):
    """
    Calculate the weights and cost matrix for the given client and facility points.

    Parameters:
    - client_points (gpd.GeoDataFrame): GeoDataFrame of client points with weights.
    - facility_points (gpd.GeoDataFrame): GeoDataFrame of facility points.
    - service_radius (float): Service radius.
    - mapbox_access_token (str): Mapbox access token.

    Returns:
    - np.array: Array of weights.
    - np.array: Cost matrix.
    """
    weights = client_points['weights'].values
    cost_matrix = calculate_cost_matrix(client_points, facility_points, service_radius, mapbox_access_token)
    return weights, cost_matrix


def setup_and_solve_mclp(cost_matrix, weights, service_radius, p_facilities):
    """
    Set up and solve the Maximal Covering Location Problem (MCLP).

    Parameters:
    - cost_matrix (np.array): Cost matrix.
    - weights (np.array): Weights for the demand points.
    - service_radius (float): Service radius.
    - p_facilities (int): Number of facilities to locate.

    Returns:
    - spopt.locate.coverage.MCLP: Solved MCLP model.
    """
    mclp = MCLP.from_cost_matrix(cost_matrix, weights, service_radius, p_facilities)
    solver = PULP_CBC_CMD()
    mclp.solve(solver)
    return mclp


def print_coverage_results(mclp):
    """
    Print the coverage results from the solved MCLP model.

    Parameters:
    - mclp (spopt.locate.coverage.MCLP): Solved MCLP model.
    """
    print(f"{mclp.perc_cov}% coverage is observed")


def create_network_with_lattice(client_points, spacing=10):
    """
    Create a network with a regular lattice centered around the extent.

    Parameters:
    - client_points (gpd.GeoDataFrame): GeoDataFrame of client points.

    Returns:
    - spaghetti.Network: Created network with a regular lattice.
    """
    minx, miny, maxx, maxy = client_points.total_bounds
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lattice = spaghetti.regular_lattice((minx, miny, maxx, maxy), spacing, exterior=True)
    ntw = spaghetti.Network(in_data=lattice)
    return ntw


def snap_observations_to_network(ntw, client_points, facility_points):
    """
    Snap observations (client and facility points) to the network.

    Parameters:
    - ntw (spaghetti.Network): Network to snap observations to.
    - client_points (gpd.GeoDataFrame): GeoDataFrame of client points.
    - facility_points (gpd.GeoDataFrame): GeoDataFrame of facility points.

    Returns:
    - gpd.GeoDataFrame: Snapped client points.
    - gpd.GeoDataFrame: Snapped facility points.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ntw.snapobservations(client_points, "clients", attribute=True)
    clients_snapped = spaghetti.element_as_gdf(ntw, pp_name="clients", snapped=True)
    clients_snapped.drop(columns=["id", "comp_label"], inplace=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ntw.snapobservations(facility_points, "facilities", attribute=True)
    facilities_snapped = spaghetti.element_as_gdf(ntw, pp_name="facilities", snapped=True)
    facilities_snapped.drop(columns=["id", "comp_label"], inplace=True)

    return clients_snapped, facilities_snapped


def visualize_results(client_points, facility_points, streets, mclp_result):
    """
    Visualize the results on a map.

    Parameters:
    - client_points (gpd.GeoDataFrame): GeoDataFrame of client points.
    - facility_points (gpd.GeoDataFrame): GeoDataFrame of facility points.
    - streets (gpd.GeoDataFrame): GeoDataFrame of streets.
    - mclp_result (spopt.locate.coverage.MCLP): Solved MCLP model.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    streets.plot(ax=ax, alpha=0.8, zorder=1, label="streets")
    facility_points.plot(
        ax=ax, color="red", zorder=2, label=f"facility candidate sites ($n$={FACILITY_COUNT})"
    )
    client_points.plot(ax=ax, color="black", label=f"clients sites ($n$={CLIENT_COUNT})")
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.show()


def perform_grid_search_on_service_radius(cost_matrix, weight_array, p_facilities):
    """
    Perform grid search on the service radius to find the best coverage.

    Parameters:
    - cost_matrix (np.array): Cost matrix.
    - weight_array (np.array): Array of weights.
    - p_facilities (int): Number of facilities to locate.

    Returns:
    - list: List of tuples containing service radius and coverage.
    """
    service_radii = range(1, 500, 50)  # Adjust the range and step size as needed
    coverage_results = []

    for service_radius in service_radii:
        mclp_from_cm = MCLP.from_cost_matrix(
            cost_matrix,
            weight_array,
            service_radius,
            p_facilities=p_facilities,
            name="mclp-network-distance"
        )
        solver = PULP_CBC_CMD()
        mclp_from_cm.solve(solver)
        coverage_results.append((service_radius, mclp_from_cm.perc_cov))

    return coverage_results


def visualize_folium_results(demand_coords, weight_array, facility_coords, mclp_result, output_file):
    """
    Visualize the MCLP results on an interactive Folium map.

    Parameters:
    - demand_coords (list): List of demand coordinates.
    - weight_array (np.array): Array of weights.
    - facility_coords (list): List of facility coordinates.
    - mclp_result (spopt.locate.coverage.MCLP): Solved MCLP model.
    - output_file (str): Path to save the map HTML file.
    """
    HILO_CENTER = [19.7074, -155.0885]
    m = folium.Map(location=HILO_CENTER, zoom_start=13)

    covered_demand_indices = set()
    for facility_clients in mclp_result.fac2cli:
        for cli_idx in facility_clients:
            if cli_idx != -1:
                covered_demand_indices.add(cli_idx)

    for idx, ((lon, lat), weight) in enumerate(zip(demand_coords, weight_array)):
        color = 'green' if idx in covered_demand_indices else 'blue'
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=f'Demand Point (weight: {weight})'
        ).add_to(m)

    for i, (lon, lat) in enumerate(facility_coords):
        if mclp_result.fac_vars[i].varValue == 1:
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.8,
                popup=f'Facility Location {i}'
            ).add_to(m)

    legend_html = '''
         <div style="position: fixed; 
                     bottom: 50px; left: 50px; width: 200px; height: 150px; 
                     border:2px solid grey; z-index:9999; font-size:14px;
                     background-color:white; opacity: 0.85;">
         &nbsp; <i class="fa fa-map-marker fa-2x" style="color:green"></i>&nbsp; Covered Demand Point<br>
         &nbsp; <i class="fa fa-map-marker fa-2x" style="color:blue"></i>&nbsp; Uncovered Demand Point<br>
         &nbsp; <i class="fa fa-map-marker fa-2x" style="color:red"></i>&nbsp; Facility Location
         </div>
         '''

    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(output_file)
