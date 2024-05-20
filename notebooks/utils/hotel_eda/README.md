
### README.md

```markdown


# Foot_Traffic_Functions.py

This Python script simplifies the handling and analysis of geospatial data, leveraging powerful libraries like GeoPandas, Fiona, and Matplotlib to enable tasks such as data fetching, transformation, spatial joins, and visualization. Designed for geospatial analysts and data scientists, it helps in processing, analyzing, and plotting geospatial data efficiently.

## Installation

This toolkit provides a set of Python functions designed to facilitate the processing and visualization of geospatial data. It utilizes libraries such as GeoPandas, Matplotlib, and Scikit-Learn to perform tasks like data validation, nearest neighbor searches, and data visualization.


```bash
pip install geopandas matplotlib scikit-learn
```

## Usage

### Importing the Toolkit

You can import the functions from the toolkit into your Python script as follows:

```python
from your_script_name import check_geodataframe, get_nearest, plot_geospatial_data
```

### Available Functions

1. **check_geodataframe(gdf, name)**
   - **Purpose**: Validates whether a provided DataFrame is a GeoDataFrame, checks for a geometry column, and validates geometries.
   - **Arguments**:
     - `gdf`: DataFrame to check.
     - `name`: Descriptive name of the DataFrame for identification in output messages.
   - **Returns**: True if the GeoDataFrame is valid, otherwise False.

2. **get_nearest(src_points, candidates, k_neighbors=1)**
   - **Purpose**: Computes the nearest neighbors for given source points from the candidate points using a spatial index.
   - **Arguments**:
     - `src_points`: Source points with geometries (GeoDataFrame).
     - `candidates`: Candidate points with geometries to search within (GeoDataFrame).
     - `k_neighbors`: Number of nearest neighbors to find.
   - **Returns**: List of tuples containing indices of nearest neighbors and their respective distances.

3. **plot_geospatial_data(apartments, buildings, connections)**
   - **Purpose**: Plots geospatial data for apartments, buildings, and their connections.
   - **Arguments**:
     - `apartments`: GeoDataFrame containing apartment locations and attributes.
     - `buildings`: GeoDataFrame containing building locations and attributes.
     - `connections`: GeoDataFrame containing line geometries connecting nearest apartment-building pairs.
   - **Returns**: None (displays a plot).

   Creating a README for your Python functions is a great way to document their purposes, usage, and requirements. Here's how you could structure the README descriptions for the functions you've developed for interacting with the RentCast API and processing geospatial data.

---

## For RentCast API & Geospatial Functions

This document provides descriptions and usage examples for a set of Python functions designed to fetch, process, and analyze geospatial data from the RentCast API.

### Functions Included

1. **fetch_api_data**
   - **Description**: Fetches data from the RentCast API given a specific endpoint URL and headers. It handles the request and returns the data as a list of dictionaries.
   - **Parameters**:
     - `url (str)`: URL to the RentCast API endpoint.
     - `headers (dict)`: Headers to include in the request, typically including the API key.
   - **Returns**:
     - `list`: A list of dictionaries containing the API response data.
   - **Example Usage**:
     ```python
     url = "https://api.rentcast.io/v1/properties?city=Hilo&state=HI&propertyType=Apartment&limit=500"
     headers = {"accept": "application/json", "X-Api-Key": "your_api_key"}
     data = fetch_api_data(url, headers)
     ```

2. **convert_to_geodataframe**
   - **Description**: Converts a list of dictionaries to a GeoDataFrame, creating a geometry column from longitude and latitude.
   - **Parameters**:
     - `data (list of dict)`: Data to convert, each dictionary represents a property.
     - `lon_col (str)`: Column name for longitude values, default 'longitude'.
     - `lat_col (str)`: Column name for latitude values, default 'latitude'.
     - `crs (str)`: Coordinate reference system to use for the GeoDataFrame, default 'EPSG:4326'.
   - **Returns**:
     - `gpd.GeoDataFrame`: A GeoDataFrame containing the data with a geometry column.
   - **Example Usage**:
     ```python
     gdf = convert_to_geodataframe(data)
     ```

3. **perform_spatial_join**
   - **Description**: Performs a spatial join between two GeoDataFrames based on a specified spatial operation like intersects or contains.
   - **Parameters**:
     - `gdf1 (gpd.GeoDataFrame)`: The first GeoDataFrame.
     - `gdf2 (gpd.GeoDataFrame)`: The second GeoDataFrame to join with the first.
     - `how (str)`: Type of join, 'left', 'right', 'inner' (default 'inner').
     - `op (str)`: Spatial operation to use, 'intersects' (default), 'contains', etc.
   - **Returns**:
     - `gpd.GeoDataFrame`: The result of the spatial join.
   - **Example Usage**:
     ```python
     joined_gdf = perform_spatial_join(gdf1, gdf2)
     ```

4. **plot_geodataframes**
   - **Description**: Plots multiple GeoDataFrames with specified colors and labels.
   - **Parameters**:
     - `gdfs (list of gpd.GeoDataFrame)`: List of GeoDataFrames to plot.
     - `colors (list of str)`: Colors for each GeoDataFrame.
     - `labels (list of str)`: Labels for each GeoDataFrame in the legend.
   - **Displays**:
     - A plot with all GeoDataFrames visualized.
   - **Example Usage**:
     ```python
     plot_geodataframes([gdf1, gdf2], ['blue', 'green'], ['API Data', 'Other Data'])
     ```



### Example

Here's how you can use the toolkit in a simple workflow:

```python
import geopandas as gpd

# Load your data
apartments = gpd.read_file('path/to/apartment_data.shp')
buildings = gpd.read_file('path/to/building_data.shp')

# Check data integrity
check_geodataframe(apartments, "Apartments")
check_geodataframe(buildings, "Buildings")

# Find nearest buildings for each apartment
nearest_buildings = get_nearest(apartments, buildings)

# Visualize the results
plot_geospatial_data(apartments, buildings, nearest_buildings)
```
