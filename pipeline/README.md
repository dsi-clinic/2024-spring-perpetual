# pipeline

This folder contains python scripts (executable from command-line).

### Running the Pipeline
Ensure that Docker Desktop has been installed [https://docs.docker.com/engine/install/].
Install Visual Studio Code with the Dev Container extension [https://code.visualstudio.com/].
Create API tokens for ...[provide links] and add to new .env file as [...].
Build and run Dev Container by running `make run-interactive` in command-line.
Create an input folder or place input files (.csv dataframes and distance matrices) into an appropriate folder in `data/` (like `data/galveston_inputs/`).
Edit `pipeline/yaml/config_inputs.yml` (in older versions,`pipeline/utils/config_inputs.ini`) to include file paths for your data, desired output locations/filenames, and CVRP solver parameters.
After the container has started, open a new terminal and run the command `python3 pipeline/main.py -h` to find out how to the command-line options for running the pipeline.
Then, run `python3 pipeline/main.py -p [provider name] -s [solver name] [city name]` to execute the pipeline.

### pipeline/utils/
This directory contains files with various functions used across multiple files.
`cfg_parser.py`: Contains functions for parsing configs
`filter_df.py`: Contains functions for filtering dataframes and distance matrices
`geometry.py`: Provides helper functions and classes related to geometries.
`google_cvrp.py`: Contains functions for solving the CVRP problem
`logger.py`: Provides customized loggers for use across the application.
`storage.py`: Utilities for reading and writing files across data stores.

### Pipeline Inputs and Outputs
Input data should be placed in an appropriate folder in `data/`.
Outputs will be generated and saved in the `data/outputs` folder.
