# data

- `boundaries/`: GeoJSON boundaries of cities. Used for testing the pipeline. Fetched from the TomTom API.

- `categories/`: JSON files listing different location categories provided by API service providers (e.g., "restaurants", "schools"). Sourced directly from the providers (i.e., Bing Maps, Google Places, TomTom, and Yelp).

- `galveston_inputs/`: Manually-curated datasets of indoor and outdoor bin locations and distances in Galveston, Texas. Created by a previous Data Clinic team and reviewed and approved by Perpetual.

- `img/`: Still images of route maps.

- `output/`: Version 2 of the outputs directory. Contains JSON metadata, data files, and HMTL and plain text visualizations of routes generated through repeated simulations. Each folder represents a new simulation run and is named according to the time at which it began in UTC.

- `outputs/`: Version 1 of the outputs directory. Will be closed and consolidated with Version 2 in the near future.

- `poi/`: Points of interest (POI) scraped from different API service providers (e.g., Google Places, Yelp) or created by local residents in design workshops hosted by Perpetual.
