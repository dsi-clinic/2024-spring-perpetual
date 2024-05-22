# Perpetual

![A screenshot of an interactive map of routes for Galveston, Texas.](/assets/galveston_map.png)

_FIGURE 1. A map of optimal routes to serve the city of Galveston, Texas. Created from a dataset of participating Foodware Using Establishments, or FUEs, and their expected volume of foodware use._

Student Team:
- Anuj Agarwal
- John Christenson
- Kaiwen Dong
- Lydia Lo
- Sarah Walker (TA)

## Background

Perpetual is a non-profit organization that partners with municipal governments, zero-waste organizations, business leagues, community groups, and other stakeholders to reduce consumption of single-use disposables. Eliminating such waste conserves resources, minimizes pollution, and slashes the cost of foodware purchase and solid waste management for cities and local businesses.

To achieve this vision, Perpetual is designing systems where customers can borrow reusable containers, cups, and utensils from anywhere they would normally purchase food and drinks (e.g., restaurants, bars, and food trucks). The customers then return that foodware to one of many outdoor collection bins.  Finally, a fleet of trucks and bicycles visits these FUEs and outdoor bins on a schedule to drop off clean foodware and/or pick up dirty foodware for washing at a local _depot_. To date, Galveston, Texas; Hilo/Hawaii County, Hawaii; Ann Arbor, Michigan; and Savannah, Georgia, have begun collaborating with Perpetual to design systems for their locales.

## Problem Statement

Designing a city-wide, reusable foodware system presents many challenges. Which outdoor bin locations are most likely to reach the greatest number of customers? How should vehicles pick up and drop off foodware at FUEs with varying demands to minimize total distance traveled, and therefore, cost and environmental impact? And finally, how can this model be easily scaled to multiple cities?

The University of Chicago Data Science Institute is tackling this problem by creating a pipeline to (1) fetch points of interest (POI) like restaurants, big box grocery stores, and parks from third-party APIs; (2) label them as potential indoor and outdoor points; and (3) then, using configurable estimates for FUE demand, vehicle carrying capacity, and number of vehicles, generate a set of foodware pickup and dropoff routes for the bins that can be shared with Perpetual as interactive maps. Finally, (4) a sensitivity analysis of the routes is performed to better understand how different parameters affect total distance traveled per truck and per cup.

This repository contains the code for the pipeline in progress. Datasets for the four partner cities and for eight additional cities and counties Perpetual recently reviewed for a federal grant are also available for testing.

## Setup

### General

The following project dependencies should already be installed after following the instructions in the Data Clinic [computer setup tutorial](https://github.com/dsi-clinic/the-clinic/blob/main/tutorials/clinic-computer-setup.md).

(1) **Docker.** Ensure that [Docker Desktop](https://docs.docker.com/engine/install/) has been installed on your local machine.

(2) **Visual Studio Code.** Install [Visual Studio Code](https://code.visualstudio.com/) for your operating system.

(3) **Make**. Confirm that `make` has been installed correctly by typing the command `make --version` in a macOS terminal or Windows Subsystem for Linux Ubuntu terminal. If the command throws an error, install the `build-essentials` package by typing `sudo apt-get install build-essential` and then reattempt the confirmation.

### Data

This quarter, we will use `.gitignore` to avoid checking in our `data` folder. This has become necessary due to the large file sizes we may encounter. (GitHub can only store files up to 100MB without purchasing additional Large File Storage.)

To retrieve data for your analysis and backend pipeline, please follow these steps:

(1) Using a terminal, navigate to the root of the repository and then create a new `data` directory by entering the command `mkdir data`.

(2) Navigate to the `Development/Data` folder in our Google Shared Drive [here](https://drive.google.com/drive/u/0/folders/1bDxGFVNr010KLQUPdrHRLhi2UA76_DOP). Download all subfolders, ignoring the `business-sales` shortcut for now, and then move those subfolders underneath the repository's `data` directory.


### Backend Pipeline Only

Later this quarter, we will start incorporating your code into the backend pipeline. For that, you will need to generate the following API keys and tokens:

(1) **Google Maps API Key.** Create an API key for [Google Maps Platform](https://developers.google.com/maps/documentation/places/web-service/get-api-key). This requires the creation of a Google Cloud Project with billing information. Once the key has been generated, restrict the key to the "Places (New)" API endpoint on the "Google Maps Platform > Credentials page".  It is also recommended to set a quota for daily API calls to avoid spending above the amount permitted by the free tier ($200/month). Finally, add the key to a new `.env` file under the root of the repository and save the value as `GOOGLE_MAPS_API_KEY=<key>`, where `<key>` refers to your key value. The file will automatically be ignored by git.

(2) **Yelp Fusion API Key.** Create an API key for [Yelp Fusion](https://docs.developer.yelp.com/docs/fusion-intro) by registering a new app on the website and agreeing to Yelp's API Terms of Use and Display.  Add the key to the `.env` file as `YELP_API_KEY=<key>`.

(3) **TomTom API Key.** Create an API key for [TomTom](https://developer.tomtom.com/knowledgebase/platform/articles/how-to-get-an-tomtom-api-key/) by registering for the TomTom Developer Portal.  Copy the key that is there (called "My first API key") and add it to the `.env` file as `TOMTOM_API_KEY=<key>`.

(4) **Microsoft Bing Maps API Key.** Create an API key for [Microsoft Bing Maps](https://learn.microsoft.com/en-us/bingmaps/getting-started/bing-maps-dev-center-help/getting-a-bing-maps-key) by signing into a Microsoft account, registering for a new Bings Map account, and navigating to the Bing Maps Dev Center. From there, go to "My Account", select "My keys", and then select the option to create a new key and fill out the required information. Click the "Create" button, copy the key, and then add it to the `.env` file as `MICROSOFT_BING_API_KEY=<key>`.

(5) **Mapbox Access Token.** Create a Mapbox account if you don't already have one and then visit your Access Tokens page to generate a JSON Web Token (JWT). (Instructions can be found [here](https://docs.mapbox.com/help/getting-started/access-tokens/).). Once you have your token, copy and paste it into the `.env` file as `MAPBOX_ACCESS_TOKEN=<token>`.


After adding these values to the `.env` file, add the following additional environment variables:

```
# General
ENV=DEV

# Padlet Mapping
PADLET_SHOULD_GENERATE=False
PADLET_HOMEPAGE_URL=https://padlet.com/
PADLET_USER_NAME=
PADLET_PASSWORD=
PADLET_API_KEY=
PADLET_TEMPLATE_NAME='Foodware System Base Template'
PADLET_TEMPLATE_INDOOR_BINS_SECTION='Indoor Bins'
PADLET_TEMPLATE_OUTDOOR_BINS_SECTION='Outdoor Bins'
```

Your mentor will explain how to populate the Padlet username, password, and API key. For now, however, the Padlet map generation is skipped in the pipeline unless `PADLET_SHOULD_GENERATE` is set to True.

## Local Development

### Running a Jupyter Notebook

All work on Jupyter notebooks should occur within the `notebooks` folder under the root of the project.  Any Python scripts imported by the notebooks should be located under the `notebooks/utils` directory.

**In VS Code.** Because VS Code provides [native support for Jupyter notebooks](https://code.visualstudio.com/docs/datascience/jupyter-notebooks), you can simply create a virtual environment using pip or conda, activate it, install Python packages within it, and then select the environment when you run a new notebook for the first time. You will also be prompted to install the Jupyter package (click yes) and trust the workspace during your first notebook execution.

**In Jupyter Lab.**  For your convenience, a Jupyter Lab server has been set up for you using Docker. To start the server, confirm that Docker Desktop is running and then enter the command `make run-notebooks` in your terminal. This will pull a Docker image created by Jupyter Docker Stacks, build a new image that includes the additional packages listed in the `notebooks/requirements.txt` file, and then run a container from the image. (Note: When building the Docker image for the first time, you may have to wait 5-10 minutes based on your Internet connection speed. Subsequent builds will use cached data and skip the download steps.) After the Docker image builds and the container has started running, navigate to `http://localhost:8888/lab` in a web browser to begin working. Files saved and created in the container are also saved in the `notebooks` and `data` folders thanks to Docker volume mounting. Once you are finished working, shut down the server by entering `Ctrl-C` in the terminal and then typing "y" for yes. To install additional Python packages, shut down the server, add the packages to the requirements file, and then re-run the `make run-notebooks` command.

### Running the Backend Pipeline

(1) After ensuring the above setup steps have been completed, run the command `make run-backend` in a terminal. Under the hood, this uses Docker Compose to run three separate Docker containers that communicate over a network: (1) a PostgreSQL database, (2) a database GUI called pgAdmin, and (3) our pipeline, which is created as a Django project. The first time that these containers are stood up,  database tables are created and the test geographies from `data/boundaries` are loaded into one of the tables.

(2) Once the pipeline server is running successfully and listening on port 8080, open a new terminal in VS Code. Run `docker container ls` to see the list of containers running.  It should look something like this, although your ids will differ:

```
CONTAINER ID   IMAGE                    COMMAND                  CREATED          STATUS          PORTS                          NAMES
a3df6682203a   perpetual_pipeline       "bash setup.sh --mig…"   33 seconds ago   Up 26 seconds   4444/tcp, 5900/tcp             perpetual-pipeline-1
6bf473345f80   dpage/pgadmin4           "/entrypoint.sh"         6 days ago       Up 28 seconds   80/tcp, 0.0.0.0:443->443/tcp   perpetual-pgadmin-1
479db5cbaf4f   postgis/postgis:13-3.3   "docker-entrypoint.s…"   6 days ago       Up 31 seconds   0.0.0.0:5432->5432/tcp         perpetual-postgis-1
```

Note the id of the container containing the pipeline (here, `a3df6682203a`). Then enter the command `docker exec -it <first few characters of container id> /bin/bash`. For example, in this case, the command could be `docker exec -it a3d /bin/bash`. This "attaches" an interactive terminal with a bash shell directly to the container so you can execute commands inside of it. Note that the terminal changes to show a root user (e.g., `root@a3df6682203a`). Run `ls` to view the files within the pipeline or data folders of the container. These folders are mounted, so any changes made to them are also reflected outside of the containers, in your VS Code file explorer.

(3) Execute the pipeline by running the Django command `python3 manage.py fetch_locations test_<city name> -p <provider1> <provider2> ... --cached`. This will load previously fetched points of interest (POI) from one or more providers from file (the `--cached` option can be omitted to fetch new locations), perform initial cleaning, and then map those locations on a Padlet map. For example, you might enter `python3 manage.py test_jersey_city -p yelp google --cached`, which loads cached POIs retrieved from Yelp and Google Places for Jersey City, New Jersey. Future pipeline commands will incorporate the routing steps completed last quarter.

(4) Shut down the containers at any time by entering the command `exit` in the terminal where the containers are running.


## File Directory

For detailed file descriptions, please visit the README within each section.

- `assets`: Media from the project.

- `notebooks`: Jupyter notebooks to step through routing algorithms and points of interest classification strategies.

- `pipeline`: Data pipeline to fetch, standardize, de-dupe, classify, and route points of interest in order to simulate a reusable foodware system.

- `web`: Experimental Next.js web application used to test the pipeline. WIP.
