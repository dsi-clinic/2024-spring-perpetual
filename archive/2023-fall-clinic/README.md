# 2023-clinic-perpetual

## Project Background

Perpetual's goal as a nonprofit is to promote sustainability in large cities by reducing single-use disposable waste.  Currently, individual restaurants may be reluctant to change from disposables to reusables due to higher costs of purchasing the foodware and maintaining in-house washing facilities. This project aims to help perpetual lower the cost of establishing a system for washing and redistributing reusable foodware to participating food and drink establishments (FUEs) throughout a given city.

Perpetual has partnered with Galveston, TX for its first launch of this system. This quarter's team was responsible for creating the first working model for the collection and distribution process in Galveston and generalizing this city’s model into a pipeline for building models in future participating cities.

## Project Goals

After a customer has returned their used foodware to a collection bin, the dirty collection bins need to be picked up and taken to a washing facility. After the foodware gets cleaned and sanitized, the bins with clean foodware need to be re-distributed back to FUE locations. Perpetual needs to plan a distribution system to be implemented in Galveston for the cycle of clean and dirty foodware. This project's goals were to: 

* Optimize and visualize routes for the first distribution system in Galveston (which includes pickup and drop-off bin service by trucks and bikes)
* Create a pipeline that replicates and generalizes the routing process.
* Record metrics of every model in a feasibility report


Our final deliverables can be found in the `output` folder (see Repository Structure below for details). Note that the implementation of the routing is open to the interpretation of Perpetual. For example, if the model specifies that seven bikes will be needed for pickup, this could mean deploying seven bikes to run one route each, it could mean deploying one bike to run seven routes back to back, or it could mean some scenario in the middle. Perpetual can refer to the feasibility report to compare different implementation scenarios for each model.

## Usage

Complete these steps to run the pipeline:
1. clone the repository to your local machine
2. open a new terminal within the Perpetual repository and enter into the repository's Docker environment (see Docker & Docker Make below)  
3. generate a unique [mapbox API token](https://docs.mapbox.com/help/glossary/access-token/) then copy and paste the token into this [utils/config_mapbox.ini](/utils/config_mapbox.ini) file
4. set all of the arguments (inputs) that will be specific to your model in the [utils/config_inputs.ini](/utils/config_inputs.ini) file.
5. on the terminal run the following code in order (allow time for each step to process):

```
cd code  
python CreateFullServiceDf.py // same process in notebook at /notebooks/create_full_service_dfs.ipynb
python GenerateDistMatrix.py
python extract_capacity_demands.py  
python extract_supplementary_info.py  
python bike_conversion.py  
python optimize_cvrp.py   
python RouteViz.py   
python complete_feasibility_report.py
```

4. Inputs will come from the `data` and `utils` folders. Outputs will be generated and saved in the `output` folder: 
* `output/data` contains all of the intermediate data frames that were converted and saved during the pipeline.
* `output/routes` will have a folder for the model that was made which will have the routes and visualizations inside. 
* `output/feasibilityfile.csv` will be updated with the metrics of the model you just made.  


### Docker

### Docker & Make

We use `docker` and `make` to run our code. There are three built-in `make` commands:

* `make build-only`: This will build the image only. It is useful for testing and making changes to the Dockerfile.
* `make run-notebooks`: This will run a jupyter server which also mounts the current directory into `\program`.
* `make run-interactive`: This will create a container (with the current directory mounted as `\program`) and loads an interactive session. 

The file `Makefile` contains information about about the specific commands that are run using when calling each `make` statement.

### Developing inside a container with VS Code

If you prefer to develop inside a container with VS Code then do the following steps. Note that this works with both regular scripts as well as jupyter notebooks.

1. Open the repository in VS Code
2. At the bottom right a window may appear that says `Folder contains a Dev Container configuration file...`. If it does, select, `Reopen in Container` and you are done. Otherwise proceed to next step. 
3. Click the blue or green rectangle in the bottom left of VS code (should say something like `><` or `>< WSL`). Options should appear in the top center of your screen. Select `Reopen in Container`.


## Repository Structure

### code
Contains all the scripts that run our code throughout the pipeline. This [README.md file](/code/README.md) describes all the code scripts.

### data
Contains our raw FUE data and the cleaned and standardized FUE data used to make the final models. This [README.md file](/data/README.md) describes all the data.

### notebooks
Contains any notebooks that were necessary for the completion of the final deliverables. Primarily, this is code work that only needed to be performed once or only performed specifically on Galveston, and therefore is not automated in the pipeline. This [README.md file](/notebooks/README.md) describes the notebooks.

### output
Contains intermediary/resulting data, routes, visualizations, and feasibility reports output by the pipeline.
In the final model made for Galveston:
* Pickup will occur 4 days out of the week by [three trucks (with route visualizations)](/output/routes/converted_truck_pickup_galv) and [seven bikes (with route visualizations)](/output/routes/converted_bike_pickup_cap9_galv). Note: the bike routes that have been chosen are conditional on bikes having a capacity of 9 totes.
* Drop-off will only occur one day out of the week using [two trucks (with route visualization)](/output/routes/converted_truck_dropoff_galv)
* [The Feasibility Report](/output/feasibilityfile.csv) saves metrics of every routing model. This can be used to determine the most optimal model to implement. The accompanying [Data Dictionary](/output/data_dictionary.md) describes the metrics contained in the feasibility report. 

### utils
Contains files with commonly used functions stored in utility files
* config_inputs.ini is where users can set their arguments before running the pipeline.

## Project Team
- Yifan Wu, (yifwu@uchicago.edu)
- Jessica Cibrian (jescib123@gmail.com)
- Sarah Walker (swalker10@uchicago.edu)
- Huanlin Dai, (daihuanlin@uchicago.edu)

