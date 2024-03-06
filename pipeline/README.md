# pipeline

This folder contains python scripts (executable from command-line).

### Running the Pipeline
Ensure that Docker Desktop has been installed [https://docs.docker.com/engine/install/].
Install Visual Studio Code with the Dev Container extension [https://code.visualstudio.com/].
Create API tokens for ...[provide links] and add to new .env file as [...].
Build and run Dev Container by running `make run-interactive` in command-line.
After the container has started, open a new terminal and run the command `python3 pipeline/main.py -h` to find out how to the command-line options for running the pipeline.
Then, run `python3 pipeline/main.py -p [provider name] -s [solver name] [city name]` to execute the pipeline.

### Running Individual Files
Complete these steps to run individual files in the pipeline:
1. Create an input folder or place input files (.csv dataframes and distance
matrices) into an appropriate folder in `data/` (like `data/galveston_inputs/`).
2. edit `pipeline/yaml/config_inputs.yml` (in older versions,
`pipeline/utils/config_inputs.ini`) to include file paths for your data,
desired output locations/filenames, and CVRP solver parameters.
3. execute `python3 [name_of_file]` in command-line to run desired files.

### pipeline/utils/
This directory contains files with various functions used across multiple files.
`cfg_parser.py`: contains functions for parsing configs
`filter_df.py`: contains functions for filtering dataframes and distance matrices
`geometry.py`: ???
`google_cvrp.py`: contains functions for solving the CVRP problem
`logger.py`: ???
`storage.py`: ???

### CVRP Solver
These files in `pipeline/`, in order of execution, are involved with solving the CVRP.
- `solve_pickups_only.py` solves for pickup-only routes vehicles can take
- `segment_pickup_loops.py` subsets dataframe and distance matrices by pickup route
- `combine_dropoffs_pickups.py` duplicates rows to combine pickup/dropoff demands in one column
- `solve_pickups_and_dropoffs.py` solves for pickup-and-dropoff routes within pickup loops
- `visualize_routes.py` visualizes routes in desired folder

### Pipeline Inputs and Outputs
Input data should be placed in the `data` folder.
Outputs will be generated and saved in the `data/outputs` folder.
