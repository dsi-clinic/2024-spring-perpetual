# notebooks
### Gurobi_cvrp.ipynb:  toy example for how to builid a basic cvrp model using gurobi optimizer.

in order to run this note book on your local machine, please follow the link https://support.gurobi.com/hc/en-us/articles/14799677517585-Getting-Started-with-Gurobi-Optimizer install Gurobi Optimizer in your labtop.

### gal_gurobi.ipynb: routing result using Galveston pickup/drop-off location.

### gurobi.py: capsulation cvrp model buidling process into function.

### visual_prepare: prepare data for visualiztion. Ouput from gurobi.py is a list of arcs, this script will grab more infomation from master dataset for visuliztion later.

### RouteViz.ipynb: notebook file from laster quater, which reads inputs from config file. The purpose of this file is to visulize all the routes csv file from pipeline/utils/config_inputs.ini ['route_dir']

### gurobi_vis: routes visulization, the output is under route_vis file map_all_routes.html
