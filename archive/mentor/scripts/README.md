#### File Descriptions

**GenerateDistMatrix.py**: This script produces a distance matrix using Mapbox Matrix API, which is an essential input for the routing algorithm. It needs an input dataset with stop locations in form of a csv file with *Latitude* and *Longitude* coordinates for each location. Run this script using the shell command `python3 GenerateDistMatrix.py <input-dataset>`.

**GetCapacityList.py**: This script creates a list of capacity of the daily number of totes that have to be picked up at each location. The order of the list is the same order as the distance matrix in order to use the same index to refer to corresponding locations in both lists. Run this script using the shell command `python3 GetCapacityList.py <input-dataset>`.

**CapacityRouting.py**: This is the main routing program which uses Google OR tools to attain the most optimal way of collecting and dropping resuable foodware in a city. This script requires the distance matrix and capacity list generated from the *GenerateDistMatrix.py* and *GetCapacityList.py* script. Run this script using the shell command `python3 CapacityRouting.py <distance-matrix> <capacity-list>`. Multiple configurations for this routing program can be attempted by altering the parameters in the 'config.ini' file in the root directory of this repository.

**BuildCapacityMap.py**: This script creates an html map which maps all the routes and locations for a city. It also displays a legend and pop-up windows when each location is clicked providing details about the location.