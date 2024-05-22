"""Functions used to process and visualize USA Structures data from FEMA.
"""

# Standard library imports
from typing import Optional, Tuple

# Third-party imports
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display, HTML
from sklearn.neighbors import BallTree


FEMA_RELEVANT_COLS = ["BUILD_ID", "OCC_CLS", "PRIM_OCC", "HEIGHT", "SQFEET", "SQMETERS"]


def preview_dataset(df: pd.DataFrame, num_rows: int = 5) -> None:
    """Summarizes a USA Structures dataset, printing output to the screen.

    Args:
        df (`pd.DataFrame`): The dataset.

        num_rows (`int`): The number of rows of the DataFrame to preview.

    Returns:
        `None`
    """
    # Display count
    display(HTML("<b>Count</b>"))
    display(f"{len(df):,} record(s) and {len(df.columns):,} column(s)")

    # List columns
    display(HTML("<b>Columns</b>"))
    display(df.columns)

    # Display null counts
    display(HTML("<b>Null Counts of Relevant Columns</b>"))
    display(df[FEMA_RELEVANT_COLS].info())

    # Describe distribution of numerical columns
    display(HTML("<b>Column Value Distributions</b>"))
    display(df[["HEIGHT", "SQFEET", "SQMETERS"]].describe())

    # Preview relevant columns
    display(HTML("<b>Preview Relevant Columns</b>"))
    display(df[FEMA_RELEVANT_COLS].head(num_rows))


def draw_building_size_plot(
    footprints_gdf: gpd.GeoDataFrame,
    rentcast_intersections_gdf: gpd.GeoDataFrame,
    city_name: str,
    size_threshold: int,
) -> None:
    """Draws a plot highlighting buildings of a given
    size against larger building footprints.

    Args:
        footprints_gdf (`gpd.GeoDataFrame`): The footprints
            to display as a backdrop.

        rentcast_intersections_gdf (`gpd.GeoDataFrame`): Rentcast
            buildings intersected with their larger building footprints
            to gain additional metadata. Expected to have "squareFootage"
            and "SQFEET" columns.

        city_name (`str`): The name of the city to display in the plot
            (e.g., "Hilo, HI").

        size_threshold (`int`): The threshold at or above which buildings
            should be displayed on the plot. For example, a threshold of
            `3000` indicates that buildings whose square footage is above
            3,000 and whose building footprints also had a square footage
            above 3,000 should be plotted.

    Returns:
        `None`
    """
    # Identify large buildings
    large_buildings_gdf = rentcast_intersections_gdf.query(
        f"(squareFootage > {size_threshold}) & (SQFEET > {size_threshold})"
    )

    # Return if none found above threhold
    if len(large_buildings_gdf) == 0:
        display(
            HTML(f"<b>No buildings found at or above threshold {size_threshold}.</b>")
        )
        return

    # Otherwise, plot the building footprints
    _, ax = plt.subplots(figsize=(8, 8))
    footprints_gdf.plot(
        ax=ax,
        color="lightgrey",
        edgecolor="black",
        alpha=0.5,
        label="Building Footprints",
    )

    # Plot the locations of large buildings
    large_buildings_gdf.plot(
        ax=ax, color="cyan", markersize=10, label="Large Buildings"
    )

    # Create custom patches
    footprint_patch = mpatches.Patch(color="lightgrey", label="Building Footprints")
    building_patch = mpatches.Patch(color="cyan", label="Large Buildings")

    # Add the custom patches to the legend
    ax.legend(handles=[footprint_patch, building_patch])

    # Add axis labels
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Add a title
    ax.set_title(
        "Large Residential Buildings Within Building Footprints "
        f"for {city_name}(Using {size_threshold} Threshold)"
    )

    # Optional: remove the axis for a cleaner look
    ax.set_axis_off()

    # Show the plot
    display(plt.show())


def buffer_geometry(series: gpd.GeoSeries, radius_in_meters: float) -> gpd.GeoSeries:
    """Draws a circular buffer around each element in a GeoSeries.

    Args:
        series (`gpd.GeoSeries`): The series to buffer.

        radius_in_meters (`float`): The length of the buffer's radius, in meters.

    Returns:
        (`gpd.GeoSeries`): The buffers.
    """
    # Determine CRS of original geometry
    temp = series.copy()
    original_crs = temp.crs
    if not original_crs:
        raise ValueError(
            "A CRS must be set to project the geometry series and calculate a buffer."
        )

    # Project series to UTM CRS to more accurately buffer geometry
    utm_crs = temp.estimate_utm_crs()
    buffer_geoseries = temp.to_crs(utm_crs).buffer(radius_in_meters)

    # Reproject to original CRS and return
    return buffer_geoseries.to_crs(original_crs)


def calculate_units(row: pd.Series) -> Optional[float]:
    """Estimates the number of units in a building based
    on its square footage and property type. The function
    uses predefined average unit areas for different property
    types to estimate the number of units. If the property type
    is not in the predefined list or the square footage is not
    provided, the function returns `None`.

    Args:
        row (`pd.Series`): A DtataFrame row expected to contain
            the building's property type and its total square footage.

    Returns:
        (`float` | `None`): The estimated number of units in the
            building if the property type is recognized and square
            footage is available. Returns `None` if the property
            type is not recognized or square footage is missing.
    """
    # Define average areas based on the building type
    average_unit_areas = {
        "Single Family": 2299,
        "Multifamily": 1046,
        "Condo": 592,
        "Apartment": 592,
        "Townhouse": 592,
        "Manufactured": 2000,
        "Land": 2000,
    }

    # Get the building type from the row
    building_type = row.get("propertyType")

    # Look up the average area for the building type
    average_area = average_unit_areas.get(building_type, None)

    # Proceed only if the average area is found and 'Squarefeet' is valid
    if average_area is not None and "SQFEET" in row and row["SQFEET"] is not None:
        return row["SQFEET"] / average_area

    return None


def get_nearest(
    src_points: gpd.GeoDataFrame, candidates: gpd.GeoDataFrame, k_neighbors: int = 1
) -> Tuple[int, int, float]:
    """
    Computes the nearest neighbors for the given source points
    from the candidate points using a spatial index.

    Args:
        src_points (`gpd.GeoDataFrame`): Source points with geometries.

        candidates (`gpd.GeoDataFrame`): Candidate points with
            geometries to search within.

        k_neighbors (`int`): Number of nearest neighbors to find.

    Returns:
        (`list` of (`int`, `int`, `float`)): A list of three-item tuples, each
            containing its index, the index of its nearest neighbor, and
            the computed distance between them.
    """
    # Create a tree from the candidate coordinates
    candidate_coords = np.array(list(zip(candidates.geometry.x, candidates.geometry.y)))
    tree = BallTree(candidate_coords, leaf_size=15, metric="haversine")

    # Query the tree for the nearest neighbors
    src_coords = np.array(list(zip(src_points.geometry.x, src_points.geometry.y)))
    distances, indices = tree.query(src_coords, k=k_neighbors)

    results = [
        (src_idx, cand_idx, dist)
        for src_idx, (cand_idx, dist) in enumerate(zip(indices, distances))
    ]
    return results


def plot_geospatial_data(
    apartments_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    connections_gdf: gpd.GeoDataFrame,
) -> None:
    """Plots geospatial data for apartments, buildings, and
    their connections. Uses color coding and markers
    to differentiate between the data sets and connections.

    Args:
        apartments_gdf (`gpd.GeoDataFrame`): Contains apartment
            locations and attributes.

        buildings_gdf (`gpd.GeoDataFrame`): Contains
             building locations and attributes.

        connections_gdf (`gpd.GeoDataFrame`): Contains
            line geometries connecting nearest
            apartment-building pairs.

    Returns:
        `None`
    """
    _, ax = plt.subplots(figsize=(10, 10))

    buildings_gdf.plot(ax=ax, color="blue", markersize=5, label="Buildings")
    apartments_gdf.plot(ax=ax, color="red", markersize=5, label="Apartments")
    connections_gdf.plot(
        ax=ax, color="grey", linestyle="--", linewidth=1, label="Connections"
    )

    plt.legend()
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Geospatial Relationships")
    plt.show()
