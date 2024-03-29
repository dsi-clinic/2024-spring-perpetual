# Pipeline

This directory contains a pipeline under development that will—

1. Fetch points of interest (POI) like restaurants, big box grocery stores, and parks from third-party APIs and web scrapes.

2. Clean and standardize the POIs and then de-dupe them by performing record linkage.

3. Label the POIs as potential indoor or outdoor points using a rule-based algorithm.

4. Run repeated simulations with different parameters to generate sets of optimal pickup and dropoff routes through the points.

5. Generate a sensitivity analysis of the routes to understand how total distance traveled per vehicle and per cup vary with the parameter values.