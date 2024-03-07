# Perpetual

![A screenshot of an interactive map of routes for Galveston, Texas.](/data/img/galveston_map.png)

_FIGURE 1. A map of optimal routes to serve the city of Galveston, Texas. Created from a dataset of participating Foodware Using Establishments, or FUEs, and their expected volume of foodware use._

## Background

Perpetual is a non-profit organization that partners with municipal governments, zero-waste organizations, business leagues, community groups, and other stakeholders to reduce consumption of single-use disposables. Eliminating such waste conserves resources, minimizes pollution, and slashes the cost of foodware purchase and solid waste management for cities and local businesses.

To achieve this vision, Perpetual is designing systems where customers can borrow reusable containers, cups, and utensils from anywhere they would normally purchase food and drinks (e.g., restaurants, bars, and food trucks). The customers then return that foodware to one of many outdoor collection bins.  Finally, a fleet of trucks and bicycles visits these FUEs and outdoor bins on a schedule to drop off clean foodware and/or pick up dirty foodware for washing at a local _depot_. To date, Galveston, Texas; Hilo/Hawaii County, Hawaii; Ann Arbor, Michigan; and Savannah, Georgia, have begun collaborating with Perpetual to design systems for their locales.

## Problem Statement

Designing a city-wide, reusable foodware system presents many challenges. Which outdoor bin locations are most likely to reach the greatest number of customers? How should vehicles pick up and drop off foodware at FUEs with varying demands to minimize total distance traveled, and therefore, cost and environmental impact? And finally, how can this model be easily scaled to multiple cities?

The University of Chicago Data Science Institute is tackling this problem by creating a pipeline to (1) fetch points of interest (POI) like restaurants, big box grocery stores, and parks from third-party APIs; (2) label them as potential indoor and outdoor points; and (3) then, using configurable estimates for FUE demand, vehicle carrying capacity, and number of vehicles, generate a set of foodware pickup and dropoff routes for the bins that can be shared with Perpetual as interactive maps. Finally, (4) a sensitivity analysis of the routes is performed to better understand how different parameters affect total distance traveled per truck and per cup.

This repository contains the code for the pipeline in progress. Datasets for the four partner cities are also available for testing.

## Setup

1. **Docker.** Ensure that [Docker Desktop](https://docs.docker.com/engine/install/) has been installed on your local machine.

2. **Visual Studio Code with Dev Containers Extension.** Install [Visual Studio Code](https://code.visualstudio.com/) and then the [Dev Containers extension](https://code.visualstudio.com/docs/devcontainers/tutorial#_install-the-extension).

3. **Google Maps API Key.** Create an API key for [Google Maps Platform](https://developers.google.com/maps/documentation/places/web-service/get-api-key). This requires the creation of a Google Cloud Project with billing information. Once the key has been generated, restrict the key to the "Places (New)" API endpoint on the "Google Maps Platform > Credentials page".  It is also recommended to set a quota for daily API calls to avoid spending above the amount permitted by the free tier ($200/month). Finally, add the key to a new `.env` file under the root of the repository and save the value as `GOOGLE_MAPS_API_KEY=<key>`, where `<key>` refers to your key value. The file will automatically be ignored by git.

4. **Yelp Fusion API Key.** Create an API key for [Yelp Fusion](https://docs.developer.yelp.com/docs/fusion-intro) by registering a new app on the website and agreeing to Yelp's API Terms of Use and Display.  Add the key to the `.env` file as `YELP_API_KEY=<key>`.

5. **TomTom API Key.** Create an API key for [TomTom](https://developer.tomtom.com/knowledgebase/platform/articles/how-to-get-an-tomtom-api-key/) by registering for the TomTom Developer Portal.  Copy the key that is there (called "My first API key") and add it to the `.env` file as `TOMTOM_API_KEY=<key>`.

6. **Microsoft Bing Maps API Key.** Create an API key for [Microsoft Bing Maps](https://learn.microsoft.com/en-us/bingmaps/getting-started/bing-maps-dev-center-help/getting-a-bing-maps-key) by signing into a Microsoft account, registering for a new Bings Map account, and navigating to the Bing Maps Dev Center. From there, go to "My Account", select "My keys", and then select the option to create a new key and fill out the required information. Click the "Create" button, copy the key, and then add it to the `.env` file as `MICROSOFT_BING_API_KEY=<key>`.

7. **Mapbox Access Token.** Create a Mapbox account if you don't already have one and then visit your Access Tokens page to generate a JSON Web Token (JWT). (Instructions can be found [here](https://docs.mapbox.com/help/getting-started/access-tokens/).). Once you have your token, copy and paste it into the `.env` file as `MAPBOX_ACCESS_TOKEN=<token>`.

## Running the Pipeline

1. Reopen the repository in VS Code after installing the Dev Containers extension.  A small modal window will pop up and prompt you to reopen the window in a container due to the presence of a `.devcontainer/devcontainer.json` file.  Select yes and then wait for the Dev Container to finish building.

2. Once the container has started, open a new terminal and run a command in the form of `python3 pipeline/main.py <city> -p <provider1> <provider2> ... -s <solver>`. The city name, `<city>`, is a required argument and must be one of `ann_arbor`, `galveston`, `hilo`, or `savannah`. The option/flag `-p` refers to the points of interest provider and must be one or more of `bing`, `google`, `tomtom`, and `yelp`, although it defaults to `yelp` only. Finally, the option `-s` refers to the name of the route optimizer and must be one of `google` (the default, representing Google OR-Tools) or `gurobi`, for the Gurobi Optimizer. For example, a command might be `python3 pipeline/main.py hilo -p yelp tomtom -s google`. For more information, type `python3 pipeline/main.py -h` for help in the terminal.

3. After executing the command, the pipeline will fetch POIs from the providers, perform initial cleaning, and then use those points to run multiple simulations according to the parameters listed in `pipeline/config.dev.yaml`. Result files are saved in an `outputs` directory and include maps of individual and all routes, route data in CSV format, plain text visualization of the routes, and JSON metadata on the simulation run.

## WIP

The POI cleaning, standardization, and indoor/outdoor bin classification steps are still incomplete. For now, cached files with indoor and outdoor bin locations are loaded within `pipeline/main.py` and used as inputs for routing. The sensitivity analysis is also incomplete, although the summary stats that are collected could be used to begin creating one.

A preliminary cleaning script has been written by Data Clinic students and saved under `pipeline/utils/clean.py`. To use it, add the paths to all the scraped POI JSON files into pipeline/utls/clean.py and then run `python3 pipeline/utils/clean.py`. This will take all the json files, and combine them into one standardized and formatted CSV file, which could be incorporated into the pipeline at a later stage.

## File Directory

- `data`: Inputs and outputs for the pipeline. For more information, consult the README in the folder.
- `notebooks`: Jupyter notebooks to step through implemented routing algorithms.
- `pipeline`: Data pipeline to fetch, standardize, de-dupe, classify, and route points of interest in order to simulate a reusable foodware system.

## Contributors

### Mentor
Launa Greer

### TA
Sarah Walker

### Students
- Jessica Cibrian
- Huanlin Dai
- Lydia Lo
- Yifan Wu
