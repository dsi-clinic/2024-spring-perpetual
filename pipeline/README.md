# pipeline

This folder contains python scripts (executable from command-line).

## Usage
Complete these steps to run the pipeline:
1. Place input files (.csv dataframes and distance matrices) in `data/`.
2. edit `pipeline/utils/config_inputs.ini` to include file paths for data, desired output locations/filenames, and CVRP solver parameters.
3. on the terminal run the following code in order (allow time for each step to process):

### pipeline/utils/
This directory contains files with various functions used across multiple files.
`cfg_parser.py`: contains a config parsing function
`filter_df.py`: contains functions for filtering dataframes and distance matrices
`geometry.py`: ???
`google_cvrp.py`: contains all the functions required to solve the CVRP problem
`logger.py`: ???
`storage.py`: ???

### CVRP Solver
These files in `pipeline/` are involved with solving the CVRP.
- `combine_dropoffs.py`
- `solve_pickups_and_dropoffs.py`
- `solve_pickups_only.py`
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

### Pipeline Inputs and Outputs
Input data should be placed in the `data` folder.
Outputs will be generated and saved in the `data/outputs` folder.
