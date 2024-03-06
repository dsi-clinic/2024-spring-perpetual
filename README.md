# Perpetual

### Motivation

Perpetual is a non-profit organization that aims to reduce the use of single-use foodware. This repository contains a pipeline that Perpetual can execute to simulate a centralized foodware reuse system for any city. To date, Galveston, Texas; Hilo, Hawaii; Ann Arbor, Michigan; and Savannah, Georgia, have partnered with Perpetual to begin designing such systems. Initial datasets have been provided for those cities for pipeline testing. 

### Background

_Foodware Service Entities (FSEs)_, also known as _Foodware Using Establishments (FUEs)_ or _indoor points_, are businesses such as restaurants, bars, and food trucks that have traditionally provided food and/or drinks in disposable containers. Perpetual has begun collaborating with local zero-waste organizations and business leagues to encourage FUEs to opt into a centralized foodware reuse system. Those that participate will receive regular deliveries of reusable cups (and in the future, other types of containers) that they can pass to customers. Customers will then deposit their used cups back at the FUE or, for their convenience, at one of several _outdoor points_ (bins) nearby. Finally, the dirty cups will be collected from the FUEs and outdoor bins and cleaned in a centralized washing facility (the _depot_) before they are redistributed back to the FUEs.

Designing a feasible centralized reuse system presents many challenges. Which outdoor bin locations would result in the highest levels of customer participation? How can vehicles like trucks and bicycles pick up and drop off foodware at FUEs with varying demans to minimize total distance traveled, and thus, cost and environmental impact? This repository uses rule-based algorithms and route optimization strategies to address this problem. The map below shows how a dataset of participating locations and their expected volume of use can be turned into a map of optimal routes to serve the city of Galveston, Texas, as an example.

![A screenshot of an interactive map of routes for Galveston, Texas.](/data/img/galveston_map.png)

#### Setup

1. **Docker.** Ensure that [Docker Desktop](https://docs.docker.com/engine/install/) has been installed on your local machine.

2. **Visual Studio Code with Dev Containers Extension.** Install [Visual Studio Code](https://code.visualstudio.com/) and then the [Dev Containers extension](https://code.visualstudio.com/docs/devcontainers/tutorial#_install-the-extension).

3. **Google Maps API Key.** Create an API key for [Google Maps Platform](https://developers.google.com/maps/documentation/places/web-service/get-api-key). This requires the creation of a Google Cloud Project with billing information. Once the key has been generated, restrict the key to the "Places (New)" API endpoint on the "Google Maps Platform > Credentials page".  It is also recommended to set a quota for daily API calls to avoid spending above the amount permitted by the free tier ($200/month). Finally, add the key to a new `.env` file under the root of the repository and save the value as `GOOGLE_MAPS_API_KEY=<key>`, where `<key>` refers to your key value. The file will automatically be ignored by git.

4. **Yelp Fusion API Key.** Create an API key for [Yelp Fusion](https://docs.developer.yelp.com/docs/fusion-intro) by registering a new app on the website and agreeing to Yelp's API Terms of Use and Display.  Add the key to the `.env` file as `YELP_API_KEY=<key>`.

5. **TomTom API Key.** Create an API key for [TomTom](https://developer.tomtom.com/knowledgebase/platform/articles/how-to-get-an-tomtom-api-key/) by registering for the TomTom Developer Portal.  Copy the key that is there (called "My first API key") and add it to the `.env` file as `TOMTOM_API_KEY=<key>`.

6. **Microsoft Bing Maps API Key.** Create an API key for [Microsoft Bing Maps](https://learn.microsoft.com/en-us/bingmaps/getting-started/bing-maps-dev-center-help/getting-a-bing-maps-key) by signing into a Microsoft account, registering for a new Bings Map account, and navigating to the Bing Maps Dev Center. From there, go to "My Account", select "My keys", and then select the option to create a new key and fill out the required information. Click the "Create" button, copy the key, and then add it to the `.env` file as `MICROSOFT_BING_API_KEY=<key>`.

7. **Mapbox API Key.** Create an API key for [Mapbox]().

### Running the Pipeline

7. Build and run Dev Container by either running `make run-interactive` on terminal or by using the VS Code Extension

8. Create an input folder or place input files (.csv dataframes and distance matrices) into an appropriate folder in `data/` (like `data/galveston_inputs/`).

9. Edit `pipeline/yaml/config_inputs.yml` (in older versions,`pipeline/utils/config_inputs.ini`) to include file paths for your data, desired output locations/filenames, and CVRP solver parameters.

7. After the container has started, open a new terminal and run the command `python3 pipeline/main.py` followed by the specific commands for the part of the pipeline you want to run. For example, if you want to run the scraping for Hilo for the Yelp API, you would do `python3 pipeline/main.py hilo -p yelp`. When you run this line, main.py runs multiple scripts within the pipeline to connect to the Yelp API and pull all the relevant business info we need for Hilo and saves it under the data/poi directory as a json file. Other parts of the pipeline have different commands and you can see those if you type `python3 pipeline/main.py -h` in terminal.

8. Once you have this file of locations for all relevant APIs you want to add the paths to all the scraped json files into pipeline/utls/clean.py and then run `python3 pipeline/utils/clean.py`. This will take all the json files, and combine them into one, standard format csv while which will then be passed to the next stage of the pipeline. 

9. For the sensitivity analysis, edit `pipeline/sensitivity_analysis.py` to point to a directory of interest. Run `python3 pipeline/sensitivity_analysis`.

### Repo Organization


### pipeline/distance/

This directory contains files related to computing distances and travel durations between geographic locations.

- `mapbox.py`: Utilities for computing distances between coordinates using the Mapbox Navigation API.

### pipeline/utils/

This directory contains files with various functions used across multiple files.

- `geometry.py`: Provides helper functions and classes related to geometries.
- `logger.py`: Provides customized loggers for use across the application.
- `storage.py`: Utilities for reading and writing files across data stores.
- `clean.py`: Cleans scraped info and transforms from json to csv format.

### pipeline/scrape/

Contains modules that call external APIs for points of interest (POI) like restaurants, schools, and parks. The fetched records will be labeled as indoor and outdoor collection bins for foodware in a future pipeline script. 

- `bing.py`: Contains skeleton file for possible integration of Bing Maps API.
- `common.py`: Defines interfaces and common classes for scraping points of interest.
- `factory.py`: Module used to generate API clients using a factory design pattern.
- `google_places.py`: Provides access to geographic locations using the Google Places API.
- `tomtom.py`: Provides access to geographic locations using the TomTom Search API.
- `yelp.py`: Provides access to geographic locations using the Yelp Fusion API.


### pipeline/routes

Contains modules used to calculate, visualize, and analyze foodware pickup and dropoff routes.

- `common.py`: Defines interfaces and common classes for running routing simulations.
- `factory.py`: Module used to generate routing solver clients using a factory design pattern.
- `google_or.py`: Route optimization algorithms provided by Google OR-Tools [https://developers.google.com/optimization].
- `gurobi.py`: Route optimization algorithms provided by Gurobi Optomizer [https://www.gurobi.com/downloads/end-user-license-agreement-academic/].
- `visualize.py`: Utilities for plotting routes on a map and representing a route in plain text.

### Contributors

#### Mentor
Launa Greer

#### TA
Sarah Walker

#### Students
- Jessica Cibrian
- Huanlin Dai
- Lydia Lo
- Yifan Wu
