"""Utilities for visualizing routes.
"""

# Standard library imports
from typing import List, Tuple, Union

# Third-party imports
import folium
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely import MultiPolygon, Polygon


def route_to_plain_text(route_id: str, route_df: pd.DataFrame) -> str:
    """Creates a diagram of route legs (e.g., "Node(1) Load(5) -> Node(6) Load(10)")
    and top-level metadata (cumulative distance and total load) using plain text.

    Args:
        route_id (`str`): A unique identifier for the route.

        route_df (`pd.DataFrame`): The DataFrame of route stops/locations.

    Returns:
        (`str`): The visual representation of the route.
    """
    route_legs = []
    for _, row in route_df.sort_values(by="Route Stop Number").iterrows():
        route_legs.append(
            f"Node({row['Original_Index']}) Load({row['Truck_Load']})"
        )

    plan_output = [f"Route #{route_id}:"]
    plan_output.append(
        f"Distance of the route: {route_df['Cumulative_Distance'].max():,}m"
    )
    plan_output.append(f"Load of the route: {route_df['Truck_Load'].iloc[0]}")
    plan_output.append(" -> \n".join(route_legs))

    return "\n".join(plan_output)


def find_bbox(coords: pd.DataFrame) -> Tuple[float, float, float, float]:
    """Given a DataFrame of coordinates (longitude and latitude), find a bounding box
    that contains all the points of interest. This function helps reduce the
    number of nodes and edges in a osmnx graph to reduce computational complexity/time.
    Padding is added to the bounding box for improved display.
    TODO: Move function to geometry utilities module and simplify.

    Args:
        coords (`pd.DataFrame`): The coordinates. Must have a column called
            "Latitude" and a column called "Longitude".

    Returns:
        ((`float`, `float`, `float`, `float`,)): A four-item tuple with
            the maximum latitude, minimum latitude, maximum longitude, and
            minimum longitude of the bounding box, respectively.
    """
    if len(coords) == 0:
        raise ValueError("find_bbox :: no coords inputted")
    n, s, e, w = [
        coords["Latitude"].iloc[0],
        coords["Latitude"].iloc[0],
        coords["Longitude"].iloc[0],
        coords["Longitude"].iloc[0],
    ]
    for i in range(len(coords)):
        longitude = coords["Longitude"].iloc[i]
        latitude = coords["Latitude"].iloc[i]
        n, s, e, w = [
            max(latitude, n),
            min(latitude, s),
            max(longitude, e),
            min(longitude, w),
        ]
    nspad = 1.5 * (n - s)
    ewpad = 1.5 * (e - w)
    return n + nspad, s - nspad, e + ewpad, w - ewpad


def calc_routes(
    graph: nx.MultiDiGraph, route_data: pd.DataFrame
) -> List[Tuple[float, float]]:
    """Takes in a graph and a set of coordinates (w/ columns
    "Longitude" and "Latitude") and returns the set of
    shortest routes between each coordinate

    Args:
        graph (`nx.MultiDigraph`): The OSMnx street network graph.

        route_data (`pd.DataFrame`): The points/stops in the route, along with
            metadata. Expected to have the columns "Longitude" and "Latitude".

    Returns:
        (`list` of (`float`, `float`,)): A list of latitude-longitude pairs
            representing the stops of the route.
    """
    # Calculate the shortest route along the OSMnx street network for each route leg
    routes = []
    for i in range(len(route_data) - 1):
        start_node = ox.nearest_nodes(
            graph,
            route_data.iloc[i]["Longitude"],
            route_data.iloc[i]["Latitude"],
        )
        end_node = ox.nearest_nodes(
            graph,
            route_data.iloc[i + 1]["Longitude"],
            route_data.iloc[i + 1]["Latitude"],
        )
        routes.append(
            nx.shortest_path(graph, start_node, end_node, weight="length")
        )

    # Convert the OSMnx routes to latitude/longitude coordinates
    final_route = []
    for route in routes:
        for point in route:
            final_route.append(
                (graph.nodes[point]["y"], graph.nodes[point]["x"])
            )

    return final_route


def add_markers(map: folium.Map, route_data: pd.DataFrame, color: str) -> None:
    """Given a Folium map, route data, and a color,
    draw markers with popups on the given map.

    Args:
        map (`folium.Map`):  The map to modify in place.

        route_data (`pd.DataFrame`): The points/stops in the route, along with
            metadata. Expected to have the columns "Name", "Longitude", "Latitude",
            "Address", "Weekly_Dropoff_Totes", "Daily_Pickup_Totes", and "pickup_type".

        color (`str`): The marker color (e.g., "red", "blue").

    Returns:
        `None`
    """
    for i in range(len(route_data)):
        # Parse row
        row = route_data.iloc[i]
        loc_name = row["Name"]
        y, x = row[["Latitude", "Longitude"]]
        address = row["Address"]
        index = route_data.index[i]
        dropoff, pickup = row[["Weekly_Dropoff_Totes", "Daily_Pickup_Totes"]]
        pickup_type = row["pickup_type"]

        # Create HTML popup
        popup_html = f"""
            Index: {index}
            <br>
            Name: {loc_name}
            <br>
            Address: {address}
            <br>
            Dropoff (weekly): {dropoff}
            <br>
            Pickup (daily): {pickup}
            <br>
            Pickup Type: {pickup_type}
        """
        popup = folium.Popup(popup_html, max_width=700)

        # Create marker with popup and add to map
        folium.Marker(
            (y, x), popup=popup, parse_html=True, icon=folium.Icon(color=color)
        ).add_to(map)


def visualize_routes(
    routes_df: pd.DataFrame,
    boundary: Union[MultiPolygon, Polygon],
    colors: List[str],
) -> List[Tuple[str, folium.Map]]:
    """Plots each route in the DataFrame on a separate map and then creates
    a composite map containing all routes. Uses the package Folium for plotting
    and the package OSMnx to draw lines between route stops using street networks
    fetched from OpenStreetMaps.

    Args:
        routes_df (`pd.DataFrame`): The route stops. Expected to have the column
            "Route," which holds the unique route identifier, as well as the
            columns "Latitude" and "Longitude".

        boundary (`shapely.MultiPolygon`|`shapely.Polygon`): The boundary
            used to specify which OSMnx street graph data should be downloaded.

        colors (`list` of `str`): The colors to use when plotting markers
            and lines on a map.

    Returns:
        (`list` of (`str`, `folium.Map`)): A list of two-item tuples, in which
            the first item is the map label/name (e.g., "route_1", "all_routes")
            and the second item is the generated map.
    """
    # Download street graph data from OpenStreetMaps
    graph = ox.graph_from_polygon(boundary, network_type="drive")

    # Initialize map of all routes
    lon, lat = [c[0] for c in boundary.centroid.coords.xy]
    center = [lat, lon]
    all_fmap = folium.Map(location=center, tiles="OpenStreetMap", zoom_start=11)

    # For each route, graph it on the composite map and create a map for itself
    i = 0
    grouped = routes_df.groupby("Route")
    route_maps = []
    for key in grouped.groups:

        # Fetch group of route stops
        route_data = grouped.get_group(key)

        # Compute bounding box for route
        coords = route_data[["Longitude", "Latitude"]]
        n, s, e, w = find_bbox(coords)

        # Skip processing if bounding box unexpectedly has a width or height of zero
        if (n - s) == 0 or (e - w) == 0:
            continue

        # Truncate graph according to bounding box of route
        galv_graph = ox.truncate.truncate_graph_bbox(
            graph,
            n,
            s,
            e,
            w,
            truncate_by_edge=False,
            retain_all=False,
            quadrat_width=0.05,
            min_num=3,
        )

        # Compute shortest paths along street network between route stops
        route = calc_routes(galv_graph, coords)

        # Initialize map for route
        fmap = folium.Map(location=center, tiles="OpenStreetMap", zoom_start=11)

        # Add markers with popups and route lines
        color = colors[(i % len(colors))]
        add_markers(fmap, route_data, "blue")
        folium.PolyLine(locations=route, color="blue").add_to(fmap)

        # Plot OSMnx graph nodes for context
        for y, x in route:
            folium.CircleMarker(
                location=[y, x], radius=2, weight=5, color="yellow"
            ).add_to(fmap)

        # Add markers with popups and route lines to composite map
        add_markers(all_fmap, route_data, color)
        folium.PolyLine(locations=route, color=color).add_to(all_fmap)

        # Store generated map in list and iterate route counter
        route_maps.append((f"route_{key}", fmap))
        i += 1

    # Append composite map to list
    route_maps.append(("all_routes", all_fmap))
    return route_maps
