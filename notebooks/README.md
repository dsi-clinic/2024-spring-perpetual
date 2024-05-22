# notebooks

Jupyter notebooks used for data analysis and simple proof-of-concepts.

## Directory

### notebooks

**01_safegraph_point_pattern_analysis.ipynb**. Analyzes the distribution of historical foot traffic across Perpetual's partner cities to (1) confirm that the distributions are non-random and (2) naively identify foot traffic "hot spots" (clusters) as potential locations for outdoor foodware collection bins. Hilo, Hawaii, is used as a case study.

**02_safegraph_visitor_routes.ipynb**. Plots travel routes from the most-visited locations to the nearest brand locations that persons tended to visit on the same day (for ex. Starbucks to the nearest Target).

**03_safegraph_correlations.ipynb**. Determines whether business sales data can be used as an appropriate proxy for foot traffic data using Hilo, Hawaii, as a case study.

**04_rentcast_buildings_eda.ipynb**. Investigates the feasability of using the RentCast API to identify highly-occupied apartments—here, operationalized as having 90 or more rooms—in any U.S. city.

**05_fema_building_footprints_eda.ipynb**. Investigates the feasibility of identifying highly-occupied apartments using the USA Structures dataset from FEMA (Federal Emergency Management Agency, Department of Homeland Security).

**06_city_hotel_correlation.ipynb**. Analyzes hotel data from Tripadvisor alongside Safegraph foot traffic patterns and Infograph business data to determine if there are any correlations between the variables.

**07_mclp_demonsration.ipynb**. Demonstrate the functionality of the Maximal Covering Location Problem (MCLP) to identify optimal bin placement locations. Hilo, Hawaii, is used as a case study.

### scripts

All scripts are located under the `utils` directory.

**constants.py**. Constants, like file paths, used throughout the notebooks and scripts.

**correlation_testing.py**. Functions used to calculate correlations with Safegraph foot traffic data.

**fema.py**. Functions used to analyze building footprints from FEMA.

**infogroup.py** Functions used to analyze business sales data from Infogroup (now Data Axle).

**logger.py**. Provides an interface for creating instances of standard loggers.

**mclp.py**. Functions used to produce an optimal solution to the Maximal Coverage Location Problem.

**point_pattern.py**. Functions used for general geospatial point pattern analyses.

**rentcast.py**. Functions used to analyze property records from the Rentcast API.

**safegraph.py**. Functions used to analyze foot traffic patterns from Safegraph.

**tripadvisor.py**. Functions used to analyze points of interest like hotels from Tripadvisor.