# pipeline

### Overview

This directory contains a pipeline under development that will—

1. Fetch points of interest (POI) like restaurants, big box grocery stores, and parks from third-party APIs.

2. Clean, standardize, and de-dupe the POIs using record linkage.

3. Label the POIs as potential indoor or outdoor points using a rule-based algorithm.

4. Run repeated simulations with different parameters to generate sets of optimal pickup and dropoff routes through the points.

5. Generate a sensitivity analysis of the routes to understand how total distance traveled per vehicle and per cup vary with the parameter values.

The pipeline is executable from a `main.py` script. Please refer to the main README for instructions. Input data should be placed in the `data` folder. Outputs will be generated and saved in the `data/outputs` folder.

### Organization

#### `pipeline/`

Scripts and configuration.

- `main.py`: The entry point for the pipeline, which can be used to run it end-to-end.
- `constants.py`: Constants used throughout the pipeline.
- `config.dev.yaml`: Contains route simulation parameters and options to load cached data to avoid unnecesary API calls and computation.

#### `pipeline/distance/`

Contains files related to computing distances and travel durations between geographic locations.

- `mapbox.py`: Utilities for computing distances between coordinates using the Mapbox Navigation API.

#### `pipeline/routes`

Contains modules used to calculate, visualize, and analyze foodware pickup and dropoff routes.

- `common.py`: Defines interfaces and common classes for running routing simulations.
- `factory.py`: Module used to generate routing solver clients using a factory design pattern.
- `google_or.py`: Route optimization algorithms provided by Google OR-Tools [https://developers.google.com/optimization].
- `gurobi.py`: Route optimization algorithms provided by Gurobi Optomizer [https://www.gurobi.com/downloads/end-user-license-agreement-academic/].
- `visualize.py`: Utilities for plotting routes on a map and representing a route in plain text.

#### `pipeline/scrape/`

Contains modules that call external APIs for points of interest (POI) like restaurants, schools, and parks. The fetched records will be labeled as indoor and outdoor collection bins for foodware in a future pipeline script. 

- `bing.py`: Contains skeleton file for possible integration of Bing Maps API.
- `common.py`: Defines interfaces and common classes for scraping points of interest.
- `factory.py`: Module used to generate API clients using a factory design pattern.
- `google_places.py`: Provides access to geographic locations using the Google Places API.
- `tomtom.py`: Provides access to geographic locations using the TomTom Search API.
- `yelp.py`: Provides access to geographic locations using the Yelp Fusion API.

#### `pipeline/tests/`

Contains unit tests, implemented with `pytest`, for pipeline modules.

- `test_distance.py`: Tests the creation of distance matrices.
- `test_geometry.py`: Tests bounding box and coordinate functions.

#### `pipeline/utils/`

Contains utilities used across the entire pipeline.

- `geometry.py`: Provides helper functions and classes related to geometries.
- `logger.py`: Provides customized loggers for use across the application.
- `storage.py`: Utilities for reading and writing files across data stores.
- `clean.py`: Cleans scraped info and transforms from json to csv format.
