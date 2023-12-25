# Perpetual

### Background:

Perpetual is a non-profit organization that aims to reduce the use of single-use foodware. This repository contains a model that can help Perpetual simulate a centralized foodware reuse system in a city.

Perpetual provided data for the city of Galveston, one of it's first partners for this project. This repository contains the dataset and an example on how to create a simulation of a centralized foodware reuse system for Galveston, TX. However, this repository can work for any city given that the relevant data is provided.

### Goal:

A centralized reuse system consists of multiple articipating Foodware-service Entities (FSEs) which consists of all locations that provide food or drinks in disposable containers. Additionally, multiple collection bins are installed throughout the city to make it convenient for people to deposit reusable containers. These containers are then washed in a centralized washing facility and redistributed back to the participating FSEs.

Planning a feasible centralized reuse system comes with a lot of challenges. This repository uses a set of techniques to find the most optimal way of approaching this problem. The map below shows how a dataset of particiating locations and their expected volume of use can be turned into a map of optimal routes to serve the city of Galveston, TX in the most feasible manner.

![The interactive map can be found in the output directory](images/galveston_map.png)


### How to use this repository:

This program can be used to create a centralized reuse system for any city. The minimal input required for this simulation is a table of locations with coordinates and expected volume of foodware use. 

This program requires a [Mapbox Access Token](https://docs.mapbox.com/help/getting-started/access-tokens/) to use Maobix API for creting the distance matrix and visualizing routes. After cloning the repository, place the API key in the `config.ini` file inside the placeholder. The config file also requires total number of vehicles for the routing simulation, the vehicle capacities, the time limit (in seconds) for the routing solver, the central coordinates for the map and file paths. 

The entire project has been divided into 3 stages. The `run_scripts.sh` file has to be run to specify which stage to execute. The 3 stages of the project include:

1. Generating inputs for route optimization
2. Running the route optimizer
3. Building the route map

### Running the simulation:

After cloning the repository, create a virtual environment and install the packages in the requirements.txt. Replace the placeholders in the `config.ini` file with your mapbox token. Modify other paramters in the config if required. To run the simulation with another city, place the dataset in the `data` directory.

On your terminal, run the command `sh run_simulation.sh`. The simulation script will require the user to specify the stage of the process to execute and the input required for each stage in the config.ini file.

##### Note:
Stage 1 requires an input dataset of participating locations and will save a distance matrix and capacity list in the `data` directory. Stage 2 requires the generated distance matrix and capacity list to produce the route list and corresponding distance list for each route. Stage 3 requires the route list and location dataset to build the capacity map in form of an html file saved in the `outputs` directory.


-----

