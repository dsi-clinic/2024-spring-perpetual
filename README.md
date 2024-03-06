# Perpetual

### Background

Perpetual is a non-profit organization that aims to reduce the use of single-use foodware. This repository contains a model that can help Perpetual simulate a centralized foodware reuse system in a city.

Perpetual provided data for the city of Galveston, one of its first partners for this project. This repository contains the dataset and an example on how to create a simulation of a centralized foodware reuse system for Galveston, TX. However, this repository can work for any city given that the relevant data is provided.

### Goal

A centralized reuse system consists of multiple articipating Foodware-service Entities (FSEs) which consists of all locations that provide food or drinks in disposable containers. Additionally, multiple collection bins are installed throughout the city to make it convenient for people to deposit reusable containers. These containers are then washed in a centralized washing facility and redistributed back to the participating FSEs.

Planning a feasible centralized reuse system comes with a lot of challenges. This repository uses a set of techniques to find the most optimal way of approaching this problem. The map below shows how a dataset of particiating locations and their expected volume of use can be turned into a map of optimal routes to serve the city of Galveston, Texas, in the most feasible manner.

![The interactive map can be found in the output directory](archive/mentor/images/galveston_map.png)


### Running the Pipeline

1. Ensure that Docker Desktop has been installed  [https://docs.docker.com/engine/install/].

2. Install Visual Studio Code with the Dev Container extension [https://code.visualstudio.com/].

3. Create API tokens for Google Places API [https://developers.google.com/maps/documentation/places/web-service/overview] and Yelp Business Search API [https://docs.developer.yelp.com/docs/fusion-intro] and add to new .env file to corresponding key pair names for each API.

4. Build and run Dev Container by either running `make run-interactive` on terminal or by using the VS Code Extension

5. Create an input folder or place input files (.csv dataframes and distance matrices) into an appropriate folder in `data/` (like `data/galveston_inputs/`).

6. Edit `pipeline/yaml/config_inputs.yml` (in older versions,`pipeline/utils/config_inputs.ini`) to include file paths for your data, desired output locations/filenames, and CVRP solver parameters.

7. After the container has started, open a new terminal and run the command `python3 pipeline/main.py` followed by the specific commands for the part of the pipeline you want to run. For example, if you want to run the scraping for Hilo for the Yelp API, you would do `python3 pipeline/main.py hilo -p yelp`. When you run this line, main.py runs multiple scripts within the pipeline to connect to the Yelp API and pull all the relevant business info we need for Hilo and saves it under the data/poi directory as a json file. Other parts of the pipeline have different commands and you can see those if you type `python3 pipeline/main.py -h` in terminal.

8. Once you have this file of locations for all relevant APIs you want to add the paths to all the scraped json files into pipeline/utls/clean.py and then run `python3 pipeline/utils/clean.py`. This will take all the json files, and combine them into one, standard format csv while which will then be passed to the next stage of the pipeline. 

9. For the sensitivity analysis, edit `pipeline/sensitivity_analysis.py` to point to a directory of interest. Run `python3 pipeline/sensitivity_analysis`.

### Repo Organization

### pipeline/utils/
This directory contains files with various functions used across multiple files.
`cfg_parser.py`: Contains functions for parsing configs
`filter_df.py`: Contains functions for filtering dataframes and distance matrices
`geometry.py`: Provides helper functions and classes related to geometries.
`google_cvrp.py`: Contains functions for solving the CVRP problem
`logger.py`: Provides customized loggers for use across the application.
`storage.py`: Utilities for reading and writing files across data stores.
`clean.py`: Cleans scraped info and transforms from json to csv format

### pipeline/scrape/
Part of the pipeline that collects new locations for bin placement 
`bing.py`: Contains skeleton file for possible integration of Bing API
`common.py`: Defines interfaces and common classes for scraping points of interest
`factory.py`: Doc of Factories used throughout the package.
`google_places.py`: Provides access to geographic locations using the Google Places API.
`tomtom.py`: Contains skeleton file for possible integration of TomTom API
`yelp.py`: Provides access to geographic locations using the Yelp Fusion API


### pipeline/routes
Contains files which are use for routing the collected points of a city
`arcgis.py`: Route optimization algorithms provided by ArcGIS.
`google_or.py`: Route optimization algorithms provided by Google OR-Tools [https://developers.google.com/optimization].
`gurobi.py`: Route optimization algorithms provided by Gurobi Optomizer [https://www.gurobi.com/downloads/end-user-license-agreement-academic/].


### Contributors
#### Mentor
Launa Greer
#### TA
Sarah Walker
#### Students
Jessica Cibrian
Huanlin Dai
Lydia Lo
Yifan Wu