"""Functions related to point pattern analyses.
"""

# Standard library imports
import os
import webbrowser
from typing import Optional, Tuple

# Third-party imports
import contextily as ctx
import folium
import geopandas as gpd
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib.patches import Ellipse
from pointpats import centrography
from shapely.geometry import MultiPoint, Point

# Application imports
from .constants import DATA_DIR


def calculate_dispersion(points: gpd.GeoSeries) -> Tuple[Point, float]:
    """Calculates the geographic dispersion of a set of points from
    their mean center. To accurately calculate the mean center
    and the distance between points, it is necessary to first
    project the points to the Universal Transverse Mercator
    (UTM) coordinate reference system (CRS). Afterwards, the mean
    center and standard distance are converted into the desired
    output CRS.

    Args:
        points (`gpd.GeoSeries`): The set of points.

    Returns:
        ((`Point`, `float`)): A two-item tuple consisting of the
            mean center as a Shapely `Point` instance and the
            standard distance from the mean center as a `float`.
    """
    # Transform the CRS to UTM
    original_crs = points.crs
    utm_crs = points.estimate_utm_crs()
    utm_points = points.copy().to_crs(utm_crs)

    # Extract x and y coordinates from the geometry since
    # centrography.std_distance expects numeric inputs
    x, y = utm_points.x, utm_points.y

    # Calculate the standard distance in the UTM projection
    std_distance = centrography.std_distance(np.vstack((x, y)).T)

    # Calculate mean center point
    multipoint_geom_utm = MultiPoint(utm_points.tolist())
    mean_center = (
        gpd.GeoSeries(data=[multipoint_geom_utm.centroid])
        .set_crs(utm_crs)
        .to_crs(original_crs)
        .iloc[0]
    )

    return mean_center, std_distance


def draw_hexbins(
    gdf: gpd.GeoDataFrame,
    lat_col: str,
    lon_col: str,
    figsize: Tuple[int, int] = (16, 12),
    zoom: int = 12,
    output_file_name: Optional[str] = None,
) -> None:
    """Renders a static map that spatially bins the points in
    the GeoDataFrame by location and then shades the bins on a
    continuous scale based on the number of points. If an output
    file name is specified, the plot is saved as a PNG file in the
    directory "data/foot-traffic/plots".

    Args:
        gdf (`gpd.GeoDataFrame`): The dataset. Expected to have
            a CRS and a geometry column with only Shapely
            POINT objects.

        lat_col (`str`): The name of the latitude column.

        lon_col (`str`): The name of the longitude column.

        figsize ((`int`, `int`)): A two-item tuple consisting
            of the length and height of the desired output figure.
            Defaults to (16, 12).

        zoom (`int`): The zoom level of the plot. Defaults to 12.

        output_file_name (`str`): The name of the optional
            output file to write. Defaults to `None`, which
            indicates that no file should be written.

    Returns:
        `None`
    """
    # Initialize plot
    _, ax = plt.subplots(figsize=figsize)

    # Generate map
    hxb = ax.hexbin(
        gdf[lon_col],
        gdf[lat_col],
        gridsize=50,
        cmap="viridis_r",
        alpha=0.4,
        linewidths=0,
    )

    # Add basemap
    ctx.add_basemap(ax, crs=gdf.crs, source=ctx.providers.CartoDB.Positron, zoom=zoom)

    # Finalize mp colorbar and axes
    plt.colorbar(hxb)
    ax.set_axis_off()

    # Write output file if indicated
    if output_file_name:
        output_dir = f"{DATA_DIR}/foot-traffic/plots"
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, output_file_name)
        plt.savefig(output_file_path)

    # Render plot
    plt.show()


def draw_scatterplot(gdf: gpd.GeoDataFrame, lat_col: str, lon_col: str) -> None:
    """Renders a static map of the points in the GeoDataFrame.

    Args:
        gdf (`gpd.GeoDataFrame`): The dataset. Expected to have
            a CRS and a geometry column with only Shapely
            POINT objects.

        lat_col (`str`): The name of the latitude column.

        lon_col (`str`): The name of the longitude column.

    Returns:
        `None`
    """
    joint_axes = sns.jointplot(x=lon_col, y=lat_col, data=gdf, s=0.5, height=10)
    ctx.add_basemap(
        joint_axes.ax_joint,
        crs=gdf.crs,
        source=ctx.providers.CartoDB.PositronNoLabels,
        alpha=1,
    )


def draw_standard_deviational_ellipse(
    points: gpd.GeoSeries,
    mean_center: Point,
    figsize: Tuple[int, int] = (16, 12),
    output_file_name: Optional[str] = None,
) -> None:
    """Plot a standard deviational ellipse for geographic
    data to evaluate dispersion. If an output file name is
    specified, the plot is saved as a PNG file in the
    directory "data/foot-traffic/plots".

    Args:
        points (`gpd.GeoSeries`): The set of points.

        mean_center (`Point`): The calculated mean center of the points.

        figsize ((`int`, `int`)): A two-item tuple consisting
            of the length and height of the desired output figure.
            Defaults to (16, 12).

        output_file_name (`str`): The name of the optional
            output file to write. Defaults to `None`, which
            indicates that no file should be written.

    Returns:
        `None`
    """
    # Compute the axes and rotation
    major, minor, rotation = centrography.ellipse([(p.x, p.y) for p in points])

    # Set up figure and axis
    _, ax = plt.subplots(1, figsize=figsize, dpi=100)

    # Plot points
    xy = (mean_center.x, mean_center.y)
    ax.scatter(points.x, points.y, s=0.5)
    ax.scatter(*xy, color="red", marker="x", label="Mean Center")
    ax.scatter(*xy, color="limegreen", marker="o", label="Median Center")

    # Construct the standard ellipse
    ellipse = Ellipse(
        xy=xy,  # center the ellipse on our mean center
        width=major * 2,  # centrography.ellipse only gives half the axis
        height=minor * 2,
        angle=np.rad2deg(rotation),
        facecolor="none",
        edgecolor="red",
        linestyle="--",
        label="Std. Ellipse",
    )
    ax.add_patch(ellipse)

    # Add legend
    ax.legend()

    # Add basemap
    ctx.add_basemap(ax, crs=points.crs, source=ctx.providers.CartoDB.Positron)

    # Write output file if indicated
    if output_file_name:
        output_dir = f"{DATA_DIR}/foot-traffic/plots"
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, output_file_name)
        plt.savefig(output_file_path)

    # Render plot
    plt.show()


def run_hdbscan(
    df: pd.DataFrame,
    min_cluster_size: int,
    min_samples: int,
    output_file_name: Optional[str] = None,
) -> Tuple[gpd.GeoDataFrame, str]:
    """Clusters the provided data using HDBSCAN clustering and
    optionally writes the results to a file. If an output
    file name is specified, the plot is saved as a PNG file in the
    directory "data/foot-traffic/output".

    Args:
        data (`pd.DataFrame`): DataFrame containing the longitude and latitude columns.

        min_cluster_size (`int`): The minimum size of clusters; no fewer than
            this number of points will form a cluster.

        min_samples (`int`): The number of samples in a neighborhood for a
            point to be considered as a core point. This helps control
            the degree of noise.

        output_file_name (`str`): The name of the optional
            output file to write. Defaults to `None`, which
            indicates that no file should be written.

    Returns:
        ((`gpd.GeoDataFrame`, `str`)): A two-item tuple consisting of
            (1) a GeoDataFrame with cluster labels and (2) the UTM
            projection of that GeoDataFrame.
    """
    # Clean dataset
    df = df.dropna(subset=["longitude", "latitude"])

    # Convert DataFrame to GeoDataFrame with longitude and latitude
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
    )

    # Convert to UTM coordinates for accurate distance measurement
    utm_crs = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(utm_crs)

    # Perform HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples
    )
    labels = clusterer.fit_predict(
        pd.DataFrame({"x": gdf.geometry.x, "y": gdf.geometry.y})
    )
    gdf["cluster"] = labels

    # Save output if a file name is specified
    if output_file_name:

        # Define the directory for saving the file
        save_directory = f"{DATA_DIR}/foot-traffic/output"
        os.makedirs(save_directory, exist_ok=True)

        # Full path for saving the file
        save_path = os.path.join(save_directory, output_file_name)

        # Save to GeoJSON
        gdf.to_file(save_path, driver="GeoJSON")

    return gdf, utm_crs


def summarize_clusters(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Summarizes the clusters in a GeoDataFrame that includes a "cluster" label column.

    Args:
        gdf (`gpd.GeoDataFrame`): A GeoDataFrame with a "cluster" column.
            The value -1 indicates noise (an outlier).

    Returns:
        (`pd.DataFrame`): A DataFrame where each row represents a cluster
            with columns summarizing the number of points in the cluster,
            the cluster label, and the centroid geometry.
    """
    # Filter out the noise points
    clusters = gdf[gdf["cluster"] != -1]

    # Handle the case of no clusters
    if clusters.empty:
        return pd.DataFrame(columns=["cluster_label", "num_points", "geometry"])

    # Group by cluster label
    grouped = clusters.groupby("cluster")

    # Create a summary DataFrame
    summary_df = pd.DataFrame(
        {
            "num_points": grouped.size(),  # Count points per cluster
            "centroid": grouped["geometry"].apply(
                lambda x: x.unary_union.centroid
            ),  # Calculate centroid
        }
    ).reset_index()

    # Rename columns appropriately
    summary_df.columns = ["cluster_label", "num_points", "geometry"]

    return summary_df


def plot_clusters(
    df: pd.DataFrame, utm_crs: str, output_fpath: Optional[str] = None
) -> None:
    """Plots the centroids of clusters on a map, annotating each centroid
    with the number of points in the cluster, and overlays this on a basemap
    for geographical context. Can optionally save the map to a file under
    "data/foot_traffic/output".

    Args:
        df (`pd.DataFrame`): A DataFrame with the columns "cluster_label",
            "num_points", and "geometry" (centroids).

        utm_crs (`str`): The CRS of the clusters..

        output_fpath (`str`): The path to the optional
            output file to write. Defaults to `None`, which
            indicates that no file should be written.

    Returns:
        `None`
    """
    # Handle empty DataFrame
    if df.empty:
        return

    # Convert to a GeoDataFrame if necessary
    if not isinstance(df, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=utm_crs)
    else:
        gdf = df

    # Convert CRS to WGS84 for Folium
    gdf = gdf.to_crs(epsg=4326)

    # Create the base map centered on the first cluster centroid
    first_lat = gdf.iloc[0]["geometry"].y
    first_lon = gdf.iloc[0]["geometry"].x
    fmap = folium.Map(location=[first_lat, first_lon], zoom_start=12)

    # Add centroids and annotations
    for _, row in gdf.iterrows():
        # Circle Marker
        folium.CircleMarker(
            location=[row["geometry"].y, row["geometry"].x],
            radius=row["num_points"] / 700,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.6,
            popup=f"Cluster {row['cluster_label']}: {row['num_points']} points",
        ).add_to(fmap)

    # Add a legend to the map with better alignment and a visual circle marker example
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 180px; height: 100px; 
                border: 2px solid grey; z-index:9999; font-size:14px;
                text-align: center; padding-top: 10px;">
                    <b>Legend</b><br>
                    Cluster Centroids <span style="color:blue;">&#11044;</span><br>
                    Circle markers are<br>
                    proportional in size<br>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    # Save or display the map
    if output_fpath:
        os.makedirs(output_fpath, exist_ok=True)

        output_fpath = (
            output_fpath if output_fpath.endswith(".html") else output_fpath + ".html"
        )
        fmap.save(output_fpath)
        webbrowser.open(f"file://{os.path.abspath(output_fpath)}")
    else:
        temp_file_path = "temp_map.html"
        fmap.save(temp_file_path)
        webbrowser.open(f"file://{os.path.abspath(temp_file_path)}")

    display(fmap)
