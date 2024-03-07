# gurobi installation

in order to run this note book on your local machine, please follow the link https://support.gurobi.com/hc/en-us/articles/14799677517585-Getting-Started-with-Gurobi-Optimizer install Gurobi Optimizer in your laptop. or to simplify the step:
    1. open the following url in your browser: https://portal.gurobi.com/iam/login/?target=https%3A%2F%2Fportal.gurobi.com%2Fiam%2Flicenses%2Flist
        create you own account using .edu email
    2. log in to your accuont, at the user portol page, click the License tab on the left bar menu -> 'Request' button at the second left bar menu -> 'WLS Academic, GENERATE NOW!' -> follow the step complete the license request.
    3. 'python -m pip install gurobipy' in your environmnet
    4. go to the gurobi login page again: https://portal.gurobi.com/iam/login/?target=https%3A%2F%2Fportal.gurobi.com%2Fiam%2Flicenses%2Flist
        in the user portal, licenses page, click the download icon on the right of the page, follow the instruction in the pop up window

### gurobi.py: capsulation cvrp model buidling process into function.

### gurobi_visual_prepare: prepare data for visualiztion. Ouput from gurobi.py is a list of arcs, this script will grab more infomation from master dataset and calculate the accumulated distance, pickup/dropoff loads for visuliztion and parameter analysis later.


