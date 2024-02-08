import ast
import fnmatch
import os

import folium
import networkx as nx
import osmnx as ox
import pandas as pd

from pipeline.utils.utils import read_cfg


def find_bbox(coords):
    """
    Given a list of coordinates (longitude and latitude),
    find a bounding box that contains all the points of interest.
    This function helps reduce the number of nodes and edges in a
    osmnx graph to reduce computational complexity/time.
    Parameters:
        coords : pd.DataFrame
    Returns:
        n, s, e, w : float
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


def osmnx_to_latlon(graph, routes):
    """
    given a route created by osmnx (node numbers), create a list of x, y
    coordinates to draw on folium

    Parameters:
        graph : osmnx graph
        routes : list of osmnx routes
                (routes are usually lists of nodes)
    Returns:
        list of (lat, lon) coordinates
    """
    final_route = []
    for route in routes:
        for point in route:
            final_route.append(
                (graph.nodes[point]["y"], graph.nodes[point]["x"])
            )
    return final_route


def calc_routes(graph, coords):
    """
    Takes in a graph and a set of coordinates (w/ columns
    "Longitude" and "Latitude") and returns the set of
    shortest routes between each coordinate

    Parameters:
        graph : osmnx graph
        coords : dataframe
    Returns:
        routes: list of routes

    """
    routes = []
    for i in range(len(coords) - 1):
        start_node = ox.nearest_nodes(
            graph, coords.iloc[i]["Longitude"], coords.iloc[i]["Latitude"]
        )
        end_node = ox.nearest_nodes(
            graph,
            coords.iloc[i + 1]["Longitude"],
            coords.iloc[i + 1]["Latitude"],
        )
        routes.append(
            nx.shortest_path(graph, start_node, end_node, weight="length")
        )

    return osmnx_to_latlon(graph, routes)


def add_markers(f_map, points, color):
    """
    given a folium map, route data (includes location names), and a color (str),
    draw markers on the given map

    Parameters:
        f_map : folium map
        route_data : pd.DataFrame
        color : str (e.g., "red", "blue")
    """

    # icon_size = 100
    for i in range(len(points)):
        row = points.iloc[i]
        loc_name = row["Name"]
        y, x = row[["Latitude", "Longitude"]]
        address = row["Address"]
        index = points.index[i]
        dropoff, pickup = row[["Weekly_Dropoff_Totes", "Daily_Pickup_Totes"]]
        pickup_type = row["pickup_type"]
        # agg_point = row["Bike Aggregation Point"]
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
                # <br>
                # Aggregation Point: {agg_point}
                # """
        popup = folium.Popup(popup_html, max_width=700)

        folium.Marker(
            (y, x), popup=popup, parse_html=True, icon=folium.Icon(color=color)
        ).add_to(f_map)

    return None


if __name__ == '__main__':
    
    cfg = read_cfg("../pipeline/utils/config_inputs.ini", "viz.route")

    place = cfg["place"]
    route_dir = cfg["route_dir"]
    latitude = float(cfg["latitude"])
    longitude = float(cfg["longitude"])
    location = [latitude, longitude]
    colors = ast.literal_eval(cfg["colors"])

    graph = ox.graph_from_place(place, network_type="drive")
    
    all_fmap = folium.Map(
        location=location, tiles="OpenStreetMap", zoom_start=11
    )

    i = 0
    for root, _, files in os.walk(route_dir):
        for filename in fnmatch.filter(files, "*.csv"):
            filepath = os.path.join(root, filename)
            if os.path.isfile(filepath):
                name = filename[:-4]
                print(f"visualize_routes :: mapping {name}")
                route_data = pd.read_csv(filepath)
                coords = route_data[["Longitude", "Latitude"]]
                
                n, s, e, w = find_bbox(coords)
                if (n-s) == 0 or (e-w) == 0:
                    print(f"visualize_routes :: map {name} has 0 as a dimension; skipping")
                    continue
                
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
    
                route = calc_routes(galv_graph, coords)
                # route = calc_routes(graph, coords)
                # route_2 = calc_routes(galv_graph2, coords2)
    
                color = colors[(i % len(colors))]
    
                fmap = folium.Map(
                    location=location, tiles="OpenStreetMap", zoom_start=11
                )
                add_markers(fmap, route_data, "blue")
                add_markers(all_fmap, route_data, color)
    
                folium.PolyLine(locations=route, color="blue").add_to(fmap)
                folium.PolyLine(locations=route, color=color).add_to(all_fmap)
    
                for y, x in route:
                    folium.CircleMarker(
                        location=[y, x], radius=2, weight=5, color="yellow"
                    ).add_to(fmap)
    
                fmap.save(route_dir + "/" "map_" + name + ".html")
                i += 1
    all_fmap.save(route_dir + "/" + "map_all_routes.html")
    print("visualize_routes :: finished visualizing all routes!")

    