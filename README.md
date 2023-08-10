# Perpetual

### Steps to take:

1. Follow the setup instructions below and ensure that the config.ini file is available in the root directory.
2. Run the file 'GetDistanceMatrix.py' using the command `python GetDistanceMatrix.py <filepath>`. This will generate a distance matrix as a .npy file with a timestamp in the data folder. 
An example command is: `python GetDistanceMatrix.py data/indoor_outdoor.csv` 


Note: I have updated the indoor_outdoor.csv file to include Moody Gardens as the starting location. Moreover, the coordinates have been rounded to 5 decimal places. 

## Setup Instructions

### Create a Virtual Environment
`python3 -m venv perpetual_env`

### Activate the Virtual Environment
- On Windows
`perpetual_env\Scripts\activate`

- On Unix or MacOS
`source perpetual_env/bin/activate`

### Install Dependencies
`pip install -r requirements.txt`

## File Documentation

### "code" folder
**arcgis_layers_processing:** contains the data cleaning .ipynb files used to reformat the data in a way that can be used for later routing, etc. after downloading a map layer from ArcGIS web apps/online maps.

**manual_data_pipeline:** contains the jupyter notebooks used to gather all data for a city. It currently has Ann Arbor as a sample but future users can change the city and reuse it.

**routing:** contains all files used for routing (MORE ON THIS LATER IN THE SUMMER).

**automated_pipeline:** contains py files that can automatically gather and output all data for a city. Input is parcel shapefile and city name, coordinates, etc.

### "data" folder
**distance_matrix:** distance matrix between all bins in a city. Used to feed into the routing algorithm.

**Galv_Bins_July_Version:** all bins (indoors and outdoors) data manually marked by Marty Miles as if July 2023. PLEASE NOTE THAT THIS IS THE MOST UP-TO-DATE DATA about collection+distribution bins and supplemental-collection bins.

**FUE_only_data_files:** clean data tables containing only FUEs info for each city.

**Non-FUE_only_data_files:** clean data tables containing non-FUE info for each city.

**ann_arbor_parcel_data:** parcel data of Ann Arbor, MI. 

**galv_parcel_data:** parcel data of Galveston, TX.

**hilo_parcel_data:** parcel data of Hilo, HI.


### "archive_23AY" folder
This is archive files from Winter 2023 (Data Clinic I) and Spring 2023 (Data Clinic II). 
