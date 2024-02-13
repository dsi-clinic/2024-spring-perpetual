# pipeline

This folder contains python scripts (executable from command-line).

## Usage

Complete these steps to run the pipeline:
1. edit the /utils/config_inputs.ini file to account for your data, desired output locations/filenames, and cvrp solver paramaeters.
2. ??
3. ??
4. ??
5. on the terminal run the following code in order (allow time for each step to process):

### /utils/
This directory contains files with various functions used across multiple files.
`cfg_parser.py`: contains a config parsing function
`filter_df.py`: contains functions for filtering dataframes and distance matrices
`geometry.py`: ???
`google_cvrp.py`: contains all the functions required to solve the CVRP problem
`logger.py`: ???
`storage.py`: ???

### CVRP Solver
These files in the /pipeline/ folder are involved with solving the CVRP.
- `combine_dropoffs.py`
- `solve_pickups_and_dropoffs.py`
- `solvce_pickups_only.py`
- `visualize_routes.py`

To run the CVRP portion, run the following from command line.
(files that have not been implemented/ported are preceded by ??)
```
??python CreateFullServiceDf.py // same process in notebook at /notebooks/create_full_service_dfs.ipynb
??python GenerateDistMatrix.py
??python extract_capacity_demands.py  
??python extract_supplementary_info.py  
??python bike_conversion.py  
python solve_pickups_only.py
python combine_dropoffs_pickups.py
python solve_pickups_and_dropoffs.py
python visualize_routes.py   
??python complete_feasibility_report.py
```

### Inputs??
Inputs will come from the `data` and `utils` folders. Outputs will be generated and saved in the `output` folder: 
* `output/data` contains all of the intermediate data frames that were converted and saved during the pipeline.
* `output/routes` will have a folder for the model that was made which will have the routes and visualizations inside. 
* `output/feasibilityfile.csv` will be updated with the metrics of the model you just made.  

