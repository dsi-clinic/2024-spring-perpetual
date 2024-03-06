# Perpetual

### Background

Perpetual is a non-profit organization that aims to reduce the use of single-use foodware. This repository contains a model that can help Perpetual simulate a centralized foodware reuse system in a city.

Perpetual provided data for the city of Galveston, one of its first partners for this project. This repository contains the dataset and an example on how to create a simulation of a centralized foodware reuse system for Galveston, TX. However, this repository can work for any city given that the relevant data is provided.

### Goal

A centralized reuse system consists of multiple articipating Foodware-service Entities (FSEs) which consists of all locations that provide food or drinks in disposable containers. Additionally, multiple collection bins are installed throughout the city to make it convenient for people to deposit reusable containers. These containers are then washed in a centralized washing facility and redistributed back to the participating FSEs.

Planning a feasible centralized reuse system comes with a lot of challenges. This repository uses a set of techniques to find the most optimal way of approaching this problem. The map below shows how a dataset of particiating locations and their expected volume of use can be turned into a map of optimal routes to serve the city of Galveston, Texas, in the most feasible manner.

![The interactive map can be found in the output directory](archive/mentor/images/galveston_map.png)


### Winter 2024 Clinic Team 

Huanlin Dai
Yifan Wu
Lydia Lo
Jessica Cibrian


### Description of Running the Repo

#### Pipeline/Scrape Files

For each file in the scrape section of the pipeline, there are specific terminal commands to run each file and scrape relevant data. For example, to run a scrape for the Yelp API for the city of Hilo,HI: you open a dev container, then, on terminal, you run the command **python3 pipeline/main.py hilo  -p yelp**. Running this for the rest of the scrapers and/or cities is just a matter of changing terminal commands and updating the YAML configuration file. 

#### Pipeline/Utils Files

The utils directory under the pipeline folder contains files and functions used across the pipeline scripts. Since these scripts contain functions used in other files in the pipeline, there's not really a need to ever run them yourself on terminal.


#### Pipeline/main.py


The main.py file within the pipeline folder is the script that helps run everything in sync, in order to actually have a working pipeline. It works through the same YAML commands but specified for whatever part of the pipeline you want to run, for example scraping or routing. 


#### Pipeline/Routes
